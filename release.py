"""Build, test, convert and upload a conda package.

For the upload step to work you have to log into your anaconda.org account
before you run the script. The steps for this are explained here:
https://conda.io/docs/user-guide/tutorials/build-pkgs.html

"""
from os.path import join
from os.path import split
from subprocess import run

from conda_build.api import build
from conda_build.api import convert


if __name__ == "__main__":
    platforms = ["osx-64", "linux-64", "win-32", "win-64"]
    built_packages = build(".", need_source_download=False)
    converted_packages = []

    for path in built_packages:
        helper, package_name = split(path)
        out_root, os = split(helper)
        pfs = [pf for pf in platforms if pf != os]
        convert(path, output_dir=out_root, platforms=pfs)
        print(
            "\n{} was converted to the following platforms: {}\n".format(
                package_name, pfs
            )
        )
        for pf in pfs:
            converted_packages.append(join(out_root, pf, package_name))

    all_packages = built_packages + converted_packages
    for package in all_packages:
        _, package_name = split(package)
        run(["anaconda", "upload", package])
        print("\n{} was uploaded to anaconda.org".format(package_name))
