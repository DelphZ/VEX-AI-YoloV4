packages = [
    ('setuptools', 'setuptools'),
    ('scipy', 'scipy'),
    ('numpy', 'numpy'),
    ('pyyaml', 'yaml'),
    ('tqdm', 'tqdm'),
    ('pillow', 'PIL'),
    ('tensorboard', 'tensorboard'),
    ('matplotlib', 'matplotlib')
]

special_packages = ['pycocotools', 'opencv-python', 'torch', 'torchvision']

print('\n=== Base Package Versions ===')
for pkg_name, import_name in packages:
    try:
        module = __import__(import_name)
        if hasattr(module, '__version__'):
            print(f'{pkg_name}: {module.__version__}')
        else:
            print(f'{pkg_name}: installed (no version info)')
    except ImportError:
        print(f'{pkg_name}: NOT installed')

print('\n=== CUDA Related Packages ===')
for pkg_name in special_packages:
    try:
        if pkg_name == 'opencv-python':
            import cv2
            print(f'opencv-python: {cv2.__version__}')
        elif pkg_name == 'torch':
            import torch
            print(f'torch: {torch.__version__}')
        elif pkg_name == 'torchvision':
            import torchvision
            print(f'torchvision: {torchvision.__version__}')
        elif pkg_name == 'pycocotools':
            import pycocotools
            from importlib.metadata import version;
            print(f'pycocotools: {version('pycocotools')}')
        else:
            __import__(pkg_name)
            print(f'{pkg_name}: installed')
    except ImportError:
        print(f'{pkg_name}: NOT installed')