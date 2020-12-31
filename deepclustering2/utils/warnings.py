import warnings
from sys import exc_info


def _warnings(*args, **kwargs):
    if len(args) > 0:
        warnings.warn(f"Received unassigned args with args: {args}.", UserWarning)
    if len(kwargs) > 0:
        kwarg_str = ", ".join([f"{k}:{v}" for k, v in kwargs.items()])
        warnings.warn(f"Received unassigned kwargs: \n{kwarg_str}", UserWarning)


def warn_on_unused_kwargs(kwargs, depth=1):
    if len(kwargs) == 0:
        pass
    else:
        frame = get_frame(depth + 2)
        name = frame.f_globals["__name__"]
        line = frame.f_lineno
        message = f"{name} at {line} Unused kwargs: \n"
        patches = []
        for k, v in kwargs.items():
            patch = f"{k}:{v}"
            patches.append(patch)

        message += "\n".join(patches)
        warnings.warn(message=message, category=RuntimeWarning)


def get_frame_fallback(n):
    try:
        raise Exception
    except Exception:
        frame = exc_info()[2].tb_frame.f_back
        for _ in range(n):
            frame = frame.f_back
        return frame


get_frame = get_frame_fallback
