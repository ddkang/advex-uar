import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
        name='advex-uar',
        version='0.0.1.dev',
        author='Daniel Kang',
        author_email='daniel.d.kang@gmail.com',
        url='https://github.com/ddkang/advex-uar/',
        packages=['advex_uar'],
        license='Apache',
)
