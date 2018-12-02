from distutils.core import setup, Extension

nwaligntokens = Extension('alignment.nwaligntokens',
                    sources = ['src/alignment/c_lib/nwaligntokens.c'])

setup (name = 'nwaligntokens',
       version = '1.0',
       package_dir={
        "": "src",
        },
       description = 'Align tokens by Needleman-Wunsch',
       ext_modules = [nwaligntokens])
