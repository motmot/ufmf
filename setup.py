from setuptools import setup, find_packages

kws=dict(
    )

setup(name='motmot.ufmf',
      description='micro-fmf (.ufmf) format library',
      version='0.2.3',
      author='Andrew Straw',
      author_email='strawman@astraw.com',
      namespace_packages = ['motmot'],
      packages = find_packages(),
      entry_points = {
    'console_scripts': ['ufmf2fmf = motmot.ufmf.ufmf2fmf:main',
                        'ufmfcat = motmot.ufmf.ufmfcat:main',
                        'playufmf = motmot.ufmf.playufmf:main',
                        'fmf2ufmf = motmot.ufmf.fmf2ufmf:main',
                        'ufmfstats = motmot.ufmf.ufmfstats:main',
                        'ufmf_reindex = motmot.ufmf.reindex:main',
                        ],
    'gui_scripts': ['fmf2ufmf-gui = motmot.ufmf.fmf2ufmf_gui:main',
                    ],
    'motmot.fview.plugins':
    'fview_ufmf_saver = motmot.ufmf.fview_plugin:UFMFFviewSaver',
    }
      )
