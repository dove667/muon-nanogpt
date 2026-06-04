import torch
from torch import Tensor


def XXT(A: Tensor, out: Tensor) -> Tensor:
    out.copy_(torch.matmul(A, A.transpose(-1, -2)))
    return out


def XTX(A: Tensor, out: Tensor) -> Tensor:
    out.copy_(torch.matmul(A.transpose(-1, -2), A))
    return out


def ba_plus_cAA(A: Tensor, alpha: float, beta: float, out: Tensor) -> Tensor:
    out.copy_(alpha * torch.matmul(A, A) + beta * A)
    return out


def make_polar_express(
    coeff_schedule: list[tuple[float, float, float]] | tuple[tuple[float, float, float], ...],
    norm_factor: float,
):
    coeffs = tuple(tuple(coeff) for coeff in coeff_schedule)

    def polar_express(
        grad_chunk: torch.Tensor,
        momentum_buffer: torch.Tensor,
        momentum_t: torch.Tensor,
        split_baddbmm: bool = False,
    ) -> torch.Tensor:
        momentum = momentum_t.to(device=grad_chunk.device, dtype=grad_chunk.dtype)
        momentum_buffer.lerp_(grad_chunk, 1 - momentum)
        g = grad_chunk.lerp_(momentum_buffer, momentum)

        x = g.bfloat16()
        is_tall = g.size(-2) > g.size(-1)
        x = x / (x.norm(dim=(-2, -1), keepdim=True) * norm_factor + 1e-6)
        x = x.contiguous()

        if is_tall:
            a_buf = torch.empty((*x.shape[:-2], x.size(-1), x.size(-1)), device=x.device, dtype=x.dtype)
            b_buf = torch.empty_like(a_buf)
            c_buf = torch.empty_like(x)
            xb_matmul = torch.bmm if x.ndim > 2 else torch.mm
            ax_plus_xb = torch.baddbmm if x.ndim > 2 else torch.addmm

            for a, b, c in coeffs:
                XTX(x, out=a_buf)
                ba_plus_cAA(a_buf, alpha=c, beta=b, out=b_buf)
                if split_baddbmm:
                    xb_matmul(x, b_buf, out=c_buf)
                    c_buf.add_(x, alpha=a)
                else:
                    ax_plus_xb(x, x, b_buf, beta=a, out=c_buf)
                x, c_buf = c_buf, x
        else:
            a_buf = torch.empty((*x.shape[:-1], x.size(-2)), device=x.device, dtype=x.dtype)
            b_buf = torch.empty_like(a_buf)
            c_buf = torch.empty_like(x)
            bx_matmul = torch.bmm if x.ndim > 2 else torch.mm
            ax_plus_bx = torch.baddbmm if x.ndim > 2 else torch.addmm

            for a, b, c in coeffs:
                XXT(x, out=a_buf)
                ba_plus_cAA(a_buf, alpha=c, beta=b, out=b_buf)
                if split_baddbmm:
                    bx_matmul(b_buf, x, out=c_buf)
                    c_buf.add_(x, alpha=a)
                else:
                    ax_plus_bx(x, b_buf, x, beta=a, out=c_buf)
                x, c_buf = c_buf, x

        return x

    return polar_express
