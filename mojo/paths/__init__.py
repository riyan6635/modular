
# Stub for mojo.paths
def _build_mojo_source_package(path):
    return path

def is_mojo_binary_package_path(path):
    return str(path).endswith('.mojopkg')

def is_mojo_source_package_path(path):
    return True
