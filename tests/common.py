import moderngl

_static = {
    'context': None,
}


def get_context() -> moderngl.Context:
    ctx = _static.get('context')

    if ctx is None:
        ctx = moderngl.create_context(standalone=True, size=(100, 100))
        _static['context'] = ctx

    return ctx
