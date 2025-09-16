import sys
import os
import torch


class LatentBlend:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "samples1": ("LATENT",),
            "samples2": ("LATENT",),
            "blend_factor": ("FLOAT", {
                "default": 0.5,
                "min": 0,
                "max": 1,
                "step": 0.01
            }),
        }, "optional": {
            "mask1": ("MASK",),
            "mask2": ("MASK",),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "blend"

    CATEGORY = "_for_testing"

    def blend(self, samples1, samples2, blend_factor:float, blend_mode: str="normal", mask1=None, mask2=None):
        try:
            import comfy
        except:
            raise ImportError("comfy is not installed")

        samples_out = samples1.copy()
        samples1 = samples1["samples"]
        samples2 = samples2["samples"]

        if samples1.shape != samples2.shape:
            samples2 = samples2.permute(0, 3, 1, 2)
            samples2 = comfy.utils.common_upscale(samples2, samples1.shape[3], samples1.shape[2], 'bicubic', crop='center')
            samples2 = samples2.permute(0, 2, 3, 1)

        # Prepare masks (BHWC broadcasting, default to ones)
        mask1_tensor = self._prepare_mask(mask1, samples1)
        mask2_tensor = self._prepare_mask(mask2, samples1)

        # Combine masks: if both provided, use their product; if one provided, use it; if none, becomes ones
        combined_mask = mask1_tensor * mask2_tensor

        samples_blended_mode = self.blend_mode(samples1, samples2, blend_mode)
        mixed = samples1 * blend_factor + samples_blended_mode * (1 - blend_factor)

        # Apply mask: keep original where mask==0, use mixed where mask>0
        samples_blended = samples1 * (1 - combined_mask) + mixed * combined_mask

        samples_out["samples"] = samples_blended
        return (samples_out,)

    def blend_mode(self, img1, img2, mode):
        if mode == "normal":
            return img2
        else:
            raise ValueError(f"Unsupported blend mode: {mode}")

    def _prepare_mask(self, mask, reference_tensor):
        """Return a BHWC mask tensor in [0,1] matching reference_tensor's shape for broadcasting.
        - If mask is None: return ones like reference (with channel 1)
        - If mask is a dict with key 'mask': extract it
        - Accept shapes [B,H,W], [B,1,H,W], [B,H,W,1], [H,W] and broadcast to [B,H,W,1]
        """
        import torch
        device = reference_tensor.device
        dtype = reference_tensor.dtype
        batch, height, width, _channels = reference_tensor.shape

        if mask is None:
            return torch.ones((batch, height, width, 1), device=device, dtype=dtype)

        # Extract raw tensor
        raw = mask.get('mask') if isinstance(mask, dict) and 'mask' in mask else mask
        if not isinstance(raw, torch.Tensor):
            # Fallback to ones if format is unknown
            return torch.ones((batch, height, width, 1), device=device, dtype=dtype)

        m = raw.to(device=device, dtype=dtype)
        # Normalize dimensionality to [B,H,W,1]
        if m.dim() == 2:  # [H,W]
            m = m.unsqueeze(0).unsqueeze(-1)  # [1,H,W,1]
        elif m.dim() == 3:
            # could be [B,H,W] or [1,H,W]
            if m.shape[0] in (1, batch) and m.shape[-1] not in (1, width):
                # assume [B,H,W]
                m = m.unsqueeze(-1)  # [B,H,W,1]
            elif m.shape[-1] in (1, width) and m.shape[0] not in (1, batch):
                # ambiguous, try treat as [H,W,C] -> reduce to [H,W,1]
                if m.shape[-1] != 1:
                    m = m.mean(dim=-1, keepdim=True)
                m = m.unsqueeze(0)  # [1,H,W,1]
            else:
                # default: [B,H,W]
                m = m.unsqueeze(-1)
        elif m.dim() == 4:
            # could be [B,1,H, W] (BCHW) or [B,H,W,1] (BHWC)
            if m.shape[1] in (1, 3, 4):  # likely BCHW
                m = m.permute(0, 2, 3, 1)  # -> [B,H,W,C]
            # ensure single channel
            if m.shape[-1] != 1:
                m = m.mean(dim=-1, keepdim=True)
        else:
            return torch.ones((batch, height, width, 1), device=device, dtype=dtype)

        # Now m is [B,H,W,1] or [1,H,W,1]
        # Resize if needed
        h, w = m.shape[1], m.shape[2]
        if (h != height) or (w != width):
            # resize using comfy's upscale (expects BCHW)
            m_bchw = m.permute(0, 3, 1, 2)
            m_bchw = comfy.utils.common_upscale(m_bchw, width, height, 'bicubic', crop='center')
            m = m_bchw.permute(0, 2, 3, 1)

        # Broadcast batch if needed
        if m.shape[0] == 1 and batch > 1:
            m = m.repeat(batch, 1, 1, 1)

        # Clamp to [0,1]
        m = m.clamp(0.0, 1.0)
        return m