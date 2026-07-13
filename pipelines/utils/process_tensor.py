import torch

def split_image(image, mask):
    """
    Convert
    (1, 3, 512, 512), (1, 1, 512, 512) --> (4, 3, 256, 256), (1, 1, 256, 256)
    """
    B, C, H, W = image.shape
    assert H % 2 == 0 and W % 2 == 0, "H and W must be even"
    h_mid = H // 2
    w_mid = W // 2

    # --- Image patches ---
    img_tl = image[:, :, :h_mid, :w_mid]  # top-left
    img_tr = image[:, :, :h_mid, w_mid:]  # top-right
    img_bl = image[:, :, h_mid:, :w_mid]  # bottom-left
    img_br = image[:, :, h_mid:, w_mid:]  # bottom-right
    img_patches = torch.cat([img_tl, img_tr, img_bl, img_br], dim=0)

    # --- Mask patches ---
    mask_tl = mask[:, :, :h_mid, :w_mid]
    mask_tr = mask[:, :, :h_mid, w_mid:]
    mask_bl = mask[:, :, h_mid:, :w_mid]
    mask_br = mask[:, :, h_mid:, w_mid:]
    mask_patches = torch.cat([mask_tl, mask_tr, mask_bl, mask_br], dim=0)

    return img_patches, mask_patches


def split_3_image(image,
                  mask,
                  true_mask,
                  patch_size= 256):
    """
    Split image, mask, and true_mask into non-overlapping patches.
    """

    B, C, H, W = image.shape

    assert H % patch_size == 0 and W % patch_size == 0, \
        "Image size must be divisible by patch_size"

    # Number of patches along height and width
    n_h = H // patch_size
    n_w = W // patch_size

    img_patches = []
    mask_patches = []
    true_mask_patches = []

    for i in range(n_h):
        for j in range(n_w):

            h_start = i * patch_size
            h_end = h_start + patch_size

            w_start = j * patch_size
            w_end = w_start + patch_size

            # Image patch
            img_patch = image[:, :, h_start:h_end, w_start:w_end]
            img_patches.append(img_patch)

            # Mask patch
            mask_patch = mask[:, :, h_start:h_end, w_start:w_end]
            mask_patches.append(mask_patch)

            # Mask2 patch
            true_mask_patch = true_mask[:, :, h_start:h_end, w_start:w_end]
            true_mask_patches.append(true_mask_patch)

    # Concatenate along batch dimension
    img_patches = torch.cat(img_patches, dim=0)
    mask_patches = torch.cat(mask_patches, dim=0)
    true_mask_patches = torch.cat(true_mask_patches, dim=0)

    return img_patches, mask_patches, true_mask_patches


def merge_image(mask, pred_mask, aligned_label, weight_mask):
    """
    Convert all masks from
    (4, 3, 256, 256) to (1, 1, 512, 512)
    """
    assert pred_mask.shape[0] == 4, "Expect 4 patches"
    _, _, H, W = pred_mask.shape
    device = pred_mask.device

    # --- Create empty canvas ---
    merged_mask = torch.zeros(1, 1, 2 * H, 2 * W, device=device)
    merged_pred = torch.zeros(1, 1, 2 * H, 2 * W, device=device)
    merged_label = torch.zeros(1, 1, 2 * H, 2 * W, device=device)
    merged_weight = torch.zeros(1, 1, 2 * H, 2 * W, device=device)

    # --- Hard stitching ---
    merged_mask[:, :, :H, :W] = mask[0]
    merged_mask[:, :, :H, W:] = mask[1]
    merged_mask[:, :, H:, :W] = mask[2]
    merged_mask[:, :, H:, W:] = mask[3]

    merged_pred[:, :, :H, :W] = pred_mask[0]  # TL
    merged_pred[:, :, :H, W:] = pred_mask[1]  # TR
    merged_pred[:, :, H:, :W] = pred_mask[2]  # BL
    merged_pred[:, :, H:, W:] = pred_mask[3]  # BR

    merged_label[:, :, :H, :W] = aligned_label[0]
    merged_label[:, :, :H, W:] = aligned_label[1]
    merged_label[:, :, H:, :W] = aligned_label[2]
    merged_label[:, :, H:, W:] = aligned_label[3]

    merged_weight[:, :, :H, :W] = weight_mask[0]
    merged_weight[:, :, :H, W:] = weight_mask[1]
    merged_weight[:, :, H:, :W] = weight_mask[2]
    merged_weight[:, :, H:, W:] = weight_mask[3]

    return merged_mask, merged_pred, merged_label, merged_weight