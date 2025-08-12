# -*- coding: utf-8 -*-

def test_import_supar():
    import supar  # noqa: F401
    from supar import Parser  # noqa: F401

    # Basic import smoke test; detailed functional tests are elsewhere
    assert True


