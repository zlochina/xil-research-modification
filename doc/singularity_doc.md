# Guide to Singularity
<!-- ## Table of contents -->
<!-- TODO -->

## Intro
This guide was created due to the need of container for AI application as such. \
Noted information here is more of self-guide rather than complete documentation for __Singularity__. If you want to learn how to use Singularity, it is recommended to follow the guide by this [link](https://docs.sylabs.io/guides/3.8/user-guide/introduction.html)

## Info
TODO list:
- [ ] Find how to build image from scratch
- [ ] Find out if optional applications could be configured to the image. - I guess just use different images.

## Quickstart

## Useful Commands
### Regular commands
* `singularity search <CONTAINER_NAME>`, which retrieves a list of images of containers corresponding to `<CONTAINER_NAME>` on the [Container Library](https://cloud.sylabs.io/library)
* `singularity pull library://lolcow` - downloads image of the container, e.g. `library://lolcow`. Pulling `docker://` images is also possible, although you are not to guaranteed to get the same image.
* `singularity build ubuntu.sif library://ubuntu` - download pre-built image from an external resource. Will also convert image to the latest SingularityCE image format. 
    * Also `build` creates images from other images or from scratch using [definition file](https://docs.sylabs.io/guides/3.8/user-guide/definition_files.html#definition-files)
* `singularity shell <CONTAINER_FILE>|<CONTAINER_URI>` - enter the shell of the container
* `singularity exec <CONTAINER_FILE> <COMMAND> <ARGS>` - executes `<COMMAND>` with specified `<ARGS>` in the provided container
    * This command could be used with `<CONTAINER_URI>` instead of `<CONTAINER_FILE>`. In such case an ephemeral container is created, where command is executed. After that the container disappears.
* `singularity run <CONTAINER_FILE>` - triggers User-defined runscript.
    * The same workflow could be triggered by `./<CONTAINER_FILE>`
    * This command could be used with `<CONTAINER_URI>` instead of `<CONTAINER_FILE>`. This creates an ephemeral ocntainer that runs and then disappears.
### Sandbox
* `singularity build --sandbox ubuntu/ library://ubuntu` - **Sandbox mode**: creates directory `ubuntu/` with an entire Ubuntu OS
    * It is possible to use commands like `shell`, `exec` and `run` with this directory instead of `<CONTAINER_FILE>`
    * `singularity exec --writable ubuntu/ touch /foo` - usage of `--writable` option to write files within the sandbox directory
    * you can use `singularity build new-sif.sif ubuntu/`, which allows to build a container from an existing container. In this case, we **convert a sandbox (directory) to the default immutable image format

## Definition files
- [runscript documentation](https://docs.sylabs.io/guides/3.8/user-guide/definition_files.html#runscript)
- [Definition file documentation](https://docs.sylabs.io/guides/3.8/user-guide/definition_files.html#definition-files)
- Definition file has header and a body. the header determines the base container, doy is further divided into sections. E.g.
```def
BootStrap: library
From: ubuntu:16.04

%post
    apt-get -y update
    apt-get -y install date cowsay lolcat

%environment
    export LC_ALL=C
    export PATH=/usr/games:$PATH

%runscript
    date | cowsay | lolcat

%labels
    Author Sylabs
```
- Building this example would be `singularity build lolcow.sif lolcow.def`, assuming this file is named `lolcow.def`
- Explanation of sections:
    1. Header tells Singularity to use a base Ubuntu 16.04 image from Container Library
    2. The `%post` section is the place to perform installations of new applications. It is exectued within the container at build time after the base OS has been installed.
    3. `%environment` section defines some environment variables that will be available to the container at runtime
    4. `%runscript` section defines actions for the container to take when it is executed by `singularity run` or other.
    5. `%labels` section is for storing custom metadata within container.
