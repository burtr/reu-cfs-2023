
## MacOS

### <a name="GitMac">Git: MacOS</a>

The Homebrew system is a bridge between the Mac and the  distribution system of 
Linux Distros. It is modelled after distro system, but it runs native on Mac.
Homebrew will give access to the entire Linux universe of free and 
open source Software.

_Terminology:_ CLI means Command Line Interface. GUI is Graphical User Interface.

### <a name="GetHomebrew">Getting Homebrew:</a>

see: https://brew.sh/

it says:
``/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"``

but that fails, first you need:
`git -C /usr/local/Homebrew/Library/Taps/homebrew/homebrew-core fetch --unshallow`


### Getting Git with Homebrew

The possibilities for getting Git on the Mac are explained on the git website:
`https://git-scm.com/download/mac`.

> `brew install git ; rehash ; git --version `

__Note:__ The semi-colon on the Unix command line just allows you to 
do on one line what you would do on many lines. The semicolon is a call to action
for the command it precedes. 

## Windows

### <a name="GitWindows">Git: Windows</a>

The possibilities for getting Git on the Mac are explained on the git website:
`https://git-scm.com/download/win`.

A git-bash is installed, which is both the git program, and a unix shell based 
on the bash shell program. 

### <a name="Cygwin">Cygwin</a>

Install Cygwin, www.cygwin.com. You can get git from cygwin and also the important
ssh program.

From the cygwin home page select setup-x86_64.exe. Take any download server, and it will
show a long list of "pending downloads". Just go ahead and install all this.

When done, repeat the visit to cygwin.com and the click on the setup exe, and this time
choose to install git and ssh. There is a selection window. Change the pulldown on the 
upper left for View to Not Installed. Search for git, and it has "skip" in the new column. 
Use the pull-down on the right to change that to a version to install.

Do the same for ssh (called openssh). The proceed to Next/Next and it installs.

Cygwin works as a custom terminal window, and a completely isolated file system branch.
The file system branch it install is a traditional Unix file system. For instance, /usr/local/bin,
and other names familiar to unix users.

When completed, you can a _Cygwin64 Terminal_. The default install leaves an icon for this on the desktop.
Inside this window, you are on what seems to be a unix machine. The window is a unix shell, and the filesystem
is laid out as is familiar to unix programer.

Check the install with the command `which ssh`, and it should return `/usr/bin/ssh`, and `which git` should return `/usr/bin/git`.

### <a name="WSL">Windows Subsystem for Linux</a>

The other possiblity is to turn on Windows long-awaited (it was part of the original design idea of Windows NT) subsystem for linux. 

1. Goto Control Panel --> Programs --> Turn Windows Features On Or Off. 
2. Enable the “Windows Subsystem for Linux” option in the list.
3. Click ok and reboot
4. After reboot install the linux distro using the Microsoft App Store.

When accomplished, you can start and Bash shell, and you will see a standard unix file hierarchy. This is a standard linux install, 
so native Ubuntu packages work. 

The windows files are found in `/mnt/c`, and if there were other drive letters, they to we be found in `/mnt`.

The original Windows NT was intended to run three "flavors" &mdash; win32, os2 and posix. Application level "OS Servers" ran, and
all operating system calls were routed through these servers, and these servers forwarded the service request to the NT Kernel. The original
NT Kernel project as a collaboration. When Microsoft found that it was better off on its own, only the win32 OS server was fully
implemented.

See the [How To Geek](https://www.howtogeek.com/249966/how-to-install-and-use-the-linux-bash-shell-on-windows-10/) for the procedure.


## Linux

### <a name="GitLinux">Git: Linux</a>

On Ubuntu,

> `sudo apt-get install git`

Note that MacOs has <a href="https://multipass.run/">Multipass</a>, a frameless Ubuntu virtual machine for Mac. It might be tricky for Jupyter, as the running Ubuntu does not know about the Mac windowing system. However, the file systems are merged, so you can work in both worls on the same file.


