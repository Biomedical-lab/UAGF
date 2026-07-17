"""Post-hoc per-branch temperature scaling.

Reference:
    Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).
    On calibration of modern neural networks. In ICML, pp. 1321-1330.
"""

import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from torch.utils.data import DataLoader


def fit_branch_temperatures(
    model,
    val_loader: DataLoader,
    device: torch.device,
    bounds: tuple = (0.1, 10.0),
) -> tuple:
    """Fit per-branch calibration temperatures on the validation set.

    Extracts logits from each branch head, then minimizes NLL w.r.t.
    a scalar temperature for each backbone independently.

    Args:
        model: Trained AdaptiveFusionModule.
        val_loader: DataLoader over validation split.
        device: Compute device.
        bounds: Search bounds for temperature optimization.

    Returns:
        Tuple (T_c, T_s, T_e) of optimized temperatures.
    """
    model.eval()
    all_logits_c, all_logits_s, all_logits_e = [], [], []
    all_labels = []

    with torch.no_grad():
        for bc, bs, be, bl in val_loader:
            bc = bc.to(device)
            bs = bs.to(device)
            be = be.to(device)

            xc_att = model.se_c(bc)
            xs_att = model.se_s(bs)
            xe_att = model.se_e(be)

            all_logits_c.append(model.head_c(xc_att).cpu())
            all_logits_s.append(model.head_s(xs_att).cpu())
            all_logits_e.append(model.head_e(xe_att).cpu())
            all_labels.append(bl.cpu())

    logits_c = torch.cat(all_logits_c)
    logits_s = torch.cat(all_logits_s)
    logits_e = torch.cat(all_logits_e)
    labels = torch.cat(all_labels)

    def _optimize_T(logits, labels):
        """Minimize NLL w.r.t. scalar temperature T."""
        def nll(T):
            return F.cross_entropy(logits / T, labels).item()
        result = minimize_scalar(nll, bounds=bounds, method="bounded")
        return result.x

    T_c = _optimize_T(logits_c, labels)
    T_s = _optimize_T(logits_s, labels)
    T_e = _optimize_T(logits_e, labels)
    return T_c, T_s, T_e
