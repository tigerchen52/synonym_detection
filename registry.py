def register(name=None, registry=None):
    def decorator(fn, registration_name=None):
        module_name = registration_name or _default_name(fn)
        if module_name in registry:
            raise LookupError(f"module {module_name} already registered.")
        registry[module_name] = fn
        return fn
    return lambda fn: decorator(fn, name)


def _default_name(obj_class):
    return obj_class.__name__
