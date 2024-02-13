from setuptools import setup

setup(
    name='segment_torch',         # How you named your package folder (MyLib)
    packages=['segment_torch'],
    version='0.0.10',      # Start with a small number and increase it with every change you make
    license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
    description="Universal implementation of the UNet architecture for image segmentation.",   # Give a short description about your library
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='László Freund',                   # Type in your name
    author_email='freundl0509@gmail.com',      # Type in your E-Mail
    url='https://github.com/lacykaltgr/agriculture-image-processing',   # Provide either the link to your github or to your website
    keywords=['unet', 'image segmentation', 'segmentation'],   # Keywords that define your package best
    install_requires=[            # I get to this in a second
        'torch',
        'numpy',
    ],

    classifiers=[
        'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Developers',      # Define that your audience are developers
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',   # Again, pick a license
        'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)