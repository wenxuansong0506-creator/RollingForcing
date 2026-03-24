from pipeline import RollingForcingTrainingPipeline
import torch.nn.functional as F
from typing import Optional, Tuple
import copy
import torch

from model.base import RollingForcingModel


class DMD(RollingForcingModel):
    def __init__(self, args, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__(args, device)
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)
        self.same_step_across_blocks = getattr(args, "same_step_across_blocks", True)
        self.num_training_frames = getattr(args, "num_training_frames", 21)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline: RollingForcingTrainingPipeline = None

        # Step 2: Initialize all dmd hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(args, "real_guidance_scale"):
            self.real_guidance_scale = args.real_guidance_scale
            self.fake_guidance_scale = args.fake_guidance_scale
        else:
            self.real_guidance_scale = args.guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)
        self.ts_schedule = getattr(args, "ts_schedule", True)
        self.ts_schedule_max = getattr(args, "ts_schedule_max", False)
        self.min_score_timestep = getattr(args, "min_score_timestep", 0)
        self.gan_loss_weight_gen = getattr(args, "gan_loss_weight_gen", 0.0)
        self.gan_loss_weight_disc = getattr(args, "gan_loss_weight_disc", 0.0)
        self.concat_time_embeddings = getattr(args, "concat_time_embeddings", False)
        self.use_relativistic_gan = getattr(args, "relativistic_discriminator", False)
        self.use_gan_loss = (self.gan_loss_weight_gen > 0.0) or (self.gan_loss_weight_disc > 0.0)

        if self.use_gan_loss:
            self.fake_score.adding_cls_branch(
                atten_dim=1536,
                num_class=getattr(args, "num_class", 1),
                time_embed_dim=1536 if self.concat_time_embeddings else 0
            )

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

    @staticmethod
    def _get_routing_log(score_model, prefix: str) -> dict:
        routing_stats = getattr(score_model, "last_routing_stats", None)
        if not routing_stats:
            return {}
        return {
            f"{prefix}_high_noise_count": routing_stats["high_noise_count"],
            f"{prefix}_low_noise_count": routing_stats["low_noise_count"],
            f"{prefix}_high_noise_ratio_actual": routing_stats["high_noise_ratio"],
            f"{prefix}_threshold": routing_stats["threshold"],
        }

    def _compute_kl_grad(
        self, noisy_image_or_video: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        timestep: torch.Tensor,
        conditional_dict: dict, unconditional_dict: dict,
        normalization: bool = True
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Compute the fake score
        _, pred_fake_image_cond = self.fake_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )
        fake_cond_log = self._get_routing_log(self.fake_score, "fake_cond_score")

        if self.fake_guidance_scale != 0.0:
            _, pred_fake_image_uncond = self.fake_score(
                noisy_image_or_video=noisy_image_or_video,
                conditional_dict=unconditional_dict,
                timestep=timestep
            )
            fake_uncond_log = self._get_routing_log(
                self.fake_score, "fake_uncond_score")
            pred_fake_image = pred_fake_image_cond + (
                pred_fake_image_cond - pred_fake_image_uncond
            ) * self.fake_guidance_scale
        else:
            pred_fake_image = pred_fake_image_cond
            fake_uncond_log = {}

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        _, pred_real_image_cond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=conditional_dict,
            timestep=timestep
        )
        real_cond_log = self._get_routing_log(self.real_score, "real_cond_score")

        _, pred_real_image_uncond = self.real_score(
            noisy_image_or_video=noisy_image_or_video,
            conditional_dict=unconditional_dict,
            timestep=timestep
        )
        real_uncond_log = self._get_routing_log(self.real_score, "real_uncond_score")

        pred_real_image = pred_real_image_cond + (
            pred_real_image_cond - pred_real_image_uncond
        ) * self.real_guidance_scale

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        grad = (pred_fake_image - pred_real_image)

        # TODO: Change the normalizer for causal teacher
        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            p_real = (estimated_clean_image_or_video - pred_real_image)
            normalizer = torch.abs(p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad = grad / normalizer
        grad = torch.nan_to_num(grad)

        return grad, {
            "dmdtrain_gradient_norm": torch.mean(torch.abs(grad)).detach(),
            "timestep": timestep.detach(),
            "high_noise_ratio": (timestep > self.score_model_timestep_threshold).float().mean().detach(),
            **fake_cond_log,
            **fake_uncond_log,
            **real_cond_log,
            **real_uncond_log,
        }

    def _sample_critic_timestep(
        self,
        image_or_video_shape,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0
    ) -> torch.Tensor:
        min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
        critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.num_frame_per_block,
            uniform_timestep=True
        )

        if self.timestep_shift > 1:
            critic_timestep = self.timestep_shift * \
                (critic_timestep / 1000) / (1 + (self.timestep_shift - 1) * (critic_timestep / 1000)) * 1000

        return critic_timestep.clamp(self.min_step, self.max_step)

    def _run_discriminator_logits(
        self,
        noisy_fake_latent: torch.Tensor,
        noisy_real_latent: torch.Tensor,
        conditional_dict: dict,
        critic_timestep: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        conditional_dict_cloned = copy.deepcopy(conditional_dict)
        conditional_dict_cloned["prompt_embeds"] = torch.concatenate(
            (conditional_dict_cloned["prompt_embeds"], conditional_dict_cloned["prompt_embeds"]), dim=0
        )

        _, _, noisy_logit = self.fake_score(
            noisy_image_or_video=torch.concatenate((noisy_fake_latent, noisy_real_latent), dim=0),
            conditional_dict=conditional_dict_cloned,
            timestep=torch.concatenate((critic_timestep, critic_timestep), dim=0),
            classify_mode=True,
            concat_time_embeddings=self.concat_time_embeddings
        )
        noisy_fake_logit, noisy_real_logit = noisy_logit.chunk(2, dim=0)
        return noisy_fake_logit, noisy_real_logit, self._get_routing_log(self.fake_score, "gan_disc_score")

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        original_latent = image_or_video

        batch_size, num_frame = image_or_video.shape[:2]

        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            min_timestep = denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
            max_timestep = denoised_timestep_from if self.ts_schedule_max and denoised_timestep_from is not None else self.num_train_timestep
            timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                num_frame,
                self.num_frame_per_block,
                uniform_timestep=True
            )

            # TODO:should we change it to `timestep = self.scheduler.timesteps[timestep]`?
            if self.timestep_shift > 1:
                timestep = self.timestep_shift * \
                    (timestep / 1000) / \
                    (1 + (self.timestep_shift - 1) * (timestep / 1000)) * 1000
            timestep = timestep.clamp(self.min_step, self.max_step)

            noise = torch.randn_like(image_or_video)
            noisy_latent = self.scheduler.add_noise(
                image_or_video.flatten(0, 1),
                noise.flatten(0, 1),
                timestep.flatten(0, 1)
            ).detach().unflatten(0, (batch_size, num_frame))

            # Step 2: Compute the KL grad
            grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=noisy_latent,
                estimated_clean_image_or_video=original_latent,
                timestep=timestep,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict
            )

        if gradient_mask is not None:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            )[gradient_mask], (original_latent.double() - grad.double()).detach()[gradient_mask], reduction="mean")
        else:
            dmd_loss = 0.5 * F.mse_loss(original_latent.double(
            ), (original_latent.double() - grad.double()).detach(), reduction="mean")
        return dmd_loss, dmd_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Unroll generator to obtain fake videos
        pred_image, gradient_mask, denoised_timestep_from, denoised_timestep_to = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent
        )

        # Step 2: Compute the DMD loss
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=pred_image,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to
        )

        gan_g_loss = torch.zeros_like(dmd_loss)
        if self.use_gan_loss and (self.gan_loss_weight_gen > 0.0) and (clean_latent is not None):
            critic_timestep = self._sample_critic_timestep(
                image_or_video_shape, denoised_timestep_from, denoised_timestep_to)
            critic_noise = torch.randn_like(pred_image)
            noisy_fake_latent = self.scheduler.add_noise(
                pred_image.flatten(0, 1),
                critic_noise.flatten(0, 1),
                critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])
            noisy_real_latent = self.scheduler.add_noise(
                clean_latent.flatten(0, 1),
                critic_noise.flatten(0, 1),
                critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])
            noisy_fake_logit, noisy_real_logit, gan_disc_routing_log = self._run_discriminator_logits(
                noisy_fake_latent=noisy_fake_latent,
                noisy_real_latent=noisy_real_latent,
                conditional_dict=conditional_dict,
                critic_timestep=critic_timestep
            )
            if not self.use_relativistic_gan:
                gan_g_loss = F.softplus(-noisy_fake_logit.float()).mean() * self.gan_loss_weight_gen
            else:
                gan_g_loss = F.softplus(-(noisy_fake_logit - noisy_real_logit).float()).mean() * self.gan_loss_weight_gen
            dmd_log_dict.update({
                "gan_g_loss": gan_g_loss.detach(),
                "gan_critic_timestep": critic_timestep.detach(),
                **gan_disc_routing_log,
            })

        total_generator_loss = dmd_loss + gan_g_loss
        dmd_log_dict.update({
            "dmd_loss": dmd_loss.detach(),
            "generator_total_loss": total_generator_loss.detach(),
        })
        return total_generator_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        # Step 1: Run generator on backward simulated noisy input
        with torch.no_grad():
            generated_image, _, denoised_timestep_from, denoised_timestep_to = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent
            )

        # Step 2: Compute the fake prediction
        critic_timestep = self._sample_critic_timestep(
            image_or_video_shape, denoised_timestep_from, denoised_timestep_to)

        critic_noise = torch.randn_like(generated_image)
        noisy_generated_image = self.scheduler.add_noise(
            generated_image.flatten(0, 1),
            critic_noise.flatten(0, 1),
            critic_timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])

        _, pred_fake_image = self.fake_score(
            noisy_image_or_video=noisy_generated_image,
            conditional_dict=conditional_dict,
            timestep=critic_timestep
        )
        fake_critic_routing_log = self._get_routing_log(
            self.fake_score, "critic_fake_score")

        # Step 3: Compute the denoising loss for the fake critic
        if self.args.denoising_loss_type == "flow":
            from utils.wan_wrapper import WanDiffusionWrapper
            flow_pred = WanDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            )
            pred_fake_noise = None
        else:
            flow_pred = None
            pred_fake_noise = self.scheduler.convert_x0_to_noise(
                x0=pred_fake_image.flatten(0, 1),
                xt=noisy_generated_image.flatten(0, 1),
                timestep=critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])

        denoising_loss = self.denoising_loss_func(
            x=generated_image.flatten(0, 1),
            x_pred=pred_fake_image.flatten(0, 1),
            noise=critic_noise.flatten(0, 1),
            noise_pred=pred_fake_noise,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=critic_timestep.flatten(0, 1),
            flow_pred=flow_pred
        )

        gan_d_loss = torch.zeros_like(denoising_loss)
        gan_disc_routing_log = {}
        noisy_fake_logit, noisy_real_logit = None, None
        if self.use_gan_loss and (self.gan_loss_weight_disc > 0.0) and (clean_latent is not None):
            noisy_real_latent = self.scheduler.add_noise(
                clean_latent.flatten(0, 1),
                critic_noise.flatten(0, 1),
                critic_timestep.flatten(0, 1)
            ).unflatten(0, image_or_video_shape[:2])
            noisy_fake_logit, noisy_real_logit, gan_disc_routing_log = self._run_discriminator_logits(
                noisy_fake_latent=noisy_generated_image,
                noisy_real_latent=noisy_real_latent,
                conditional_dict=conditional_dict,
                critic_timestep=critic_timestep
            )
            if not self.use_relativistic_gan:
                gan_d_loss = (
                    F.softplus(-noisy_real_logit.float()).mean() +
                    F.softplus(noisy_fake_logit.float()).mean()
                ) * self.gan_loss_weight_disc
            else:
                gan_d_loss = F.softplus(-(noisy_real_logit - noisy_fake_logit).float()).mean() * self.gan_loss_weight_disc

        total_critic_loss = denoising_loss + gan_d_loss

        # Step 5: Debugging Log
        critic_log_dict = {
            "critic_timestep": critic_timestep.detach(),
            "critic_high_noise_ratio": (critic_timestep > self.score_model_timestep_threshold).float().mean().detach(),
            "denoising_loss": denoising_loss.detach(),
            "gan_d_loss": gan_d_loss.detach(),
            "critic_total_loss": total_critic_loss.detach(),
            **fake_critic_routing_log,
            **gan_disc_routing_log,
        }
        if noisy_real_logit is not None and noisy_fake_logit is not None:
            critic_log_dict.update({
                "noisy_real_logit": noisy_real_logit.detach(),
                "noisy_fake_logit": noisy_fake_logit.detach(),
            })

        return total_critic_loss, critic_log_dict
