import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
        name='advex-uar',
        version='0.0.5.dev',
        author='Daniel Kang',
        author_email='daniel.d.kang@gmail.com',
        url='https://github.com/ddkang/advex-uar/',
        packages=setuptools.find_packages(),
        license='Apache',
)
