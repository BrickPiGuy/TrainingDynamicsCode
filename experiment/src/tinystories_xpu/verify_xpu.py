from __future__ import annotations

def main() -> None:
    try:
        import torch
    except Exception as exc:
        print(f"torch import failed: {exc}")
        raise SystemExit(1) from exc

    print(f"torch: {torch.__version__}")
    has_xpu = hasattr(torch, "xpu")
    print(f"torch.xpu present: {has_xpu}")
    if not has_xpu:
        raise SystemExit(1)

    available = torch.xpu.is_available()
    print(f"torch.xpu available: {available}")
    if not available:
        raise SystemExit(1)

    count = torch.xpu.device_count()
    print(f"xpu device count: {count}")
    for index in range(count):
        try:
            print(f"xpu:{index}: {torch.xpu.get_device_name(index)}")
        except Exception:
            print(f"xpu:{index}: unknown")

    x = torch.randn(512, 512, device="xpu")
    y = x @ x.T
    torch.xpu.synchronize()
    print(f"matrix multiply ok: shape={tuple(y.shape)}, dtype={y.dtype}, device={y.device}")


if __name__ == "__main__":
    main()
