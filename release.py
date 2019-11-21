"""Build, test, convert and upload a conda package.

For the upload step to work you have to log into your anaconda.org account
before you run the script. The steps for this are explained here:
https://conda.io/docs/user-guide/tutorials/build-pkgs.html

"""
import shutil
import subprocess
from os.path import join
from os.path import split
from pathlib import Path

import click
from conda_build.api import build
from conda_build.api import convert


CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}

TEMPORARY_FOLDERS = [
    Path("documentation", "_build"),
    Path("documentation", "_generated"),
] + list(Path(".").glob("**/__pycache__"))


@click.group(context_settings=CONTEXT_SETTINGS, chain=True, invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """Build, convert and upload a conda package."""
    ctx.invoke(clean)
    ctx.invoke(build_convert_upload)


@cli.command()
def clean():
    """Clean the package folder from temporary files and folders."""
    click.secho("-" * 88)
    click.secho("Cleaning - Start\n")
    click.secho(
        "Unnecessary files in the repository can cause errors while building the\n"
        "building the package. Check for some known issues.\n"
    )

    # Check for environments in .tox.
    tox_envs = list(Path(".", ".tox").glob("*"))
    if tox_envs and click.confirm(
        "Do you want to remove all tests environments under .tox?"
    ):
        for path in tox_envs:
            subprocess.run(f"conda env remove -p {path}", shell=True)
        shutil.rmtree(".tox")

    # Check for temporary files and folders which can be deleted.
    for path in TEMPORARY_FOLDERS:
        if path.is_dir():
            shutil.rmtree(path)
        elif path.is_file():
            path.unlink()
        else:
            pass

    # Check for uncommitted and untracked files.
    files = subprocess.run(
        "git status --porcelain", shell=True, stdout=subprocess.PIPE
    ).stdout.decode("utf-8")
    if files:
        click.secho(
            "There are some uncommitted or untracked files. Please manually clean \n"
            "them up or move them somewhere else before proceeding.\n",
            color="yellow",
        )
        click.secho(files, color="yellow")
        raise click.Abort

    click.secho("Cleaning - End")
    click.secho("-" * 88 + "\n")


@cli.command()
def build_convert_upload():
    platforms = ["osx-64", "linux-32", "linux-64", "win-32", "win-64"]
    built_packages = build(".", need_source_download=False)
    converted_packages = []

    for path in built_packages:
        helper, package_name = split(path)
        out_root, os = split(helper)
        pfs = [pf for pf in platforms if pf != os]
        convert(path, output_dir=out_root, platforms=pfs)
        for pf in pfs:
            converted_packages.append(join(out_root, pf, package_name))

    all_packages = built_packages + converted_packages
    for package in all_packages:
        _, package_name = split(package)
        subprocess.run(
            ["anaconda", "upload", "--force", "--user", "OpenSourceEconomics", package]
        )


if __name__ == "__main__":
    cli()
