### SSH Public Key

Your Cane ID is your name you received at the Cane ID website. To manage or reset your password, visit https://caneidhelp.miami.edu/caneid/.

We refer to the password you used to create this account, as the Cane ID password. You have recieved other passwords for other machines,
such as the lab machines and for Pegasus. 

To help alleiviate the burden of passwords and other troublesome identifiers, as well as to enhance security, public key authentication is preferred
for ssh. It requires you create a public/private key pair using the command line program `ssh-keygen`. You can use it without parameters.
It will prompt for the name of the file to create. I suggesst `id_rsa_thoreau`, although this is not critical. 

The program will create two files: `id_rsa_thoreau` and `id_rsa_thoreau.pub`. The first contains the private key and <u>must</u> be kept secret.
The .pub file is the public key and you can share that freely. 

__public key in a nutshell__

The public key system will be a conversation between the public key holder and 
the private key holder during which the public key holder becomes convinced that the counter-party  knows the contents of the private key. 
However, this conversation reveals nothing about the private key to any party, not to an eavesdropper not even to the holder of the public key.

__Part 1__

My suggestion is that you create the key pair on thoreau, and you must do so in the directory `~/.ssh`.

1. logon onto triton. use your cane id and your cane id password
2. `mkdir .ssh`
3. `cd .ssh`
4. `ssh-keygen` (at the prompt: `id_rsa_thoreau`, and the pass phrase prompt, just return (no pass phrase), and confirm return)
5. `cat id_rsa_thoreau >> authorized_keys`
6. logout

Now transfer `id_rsa_thoreau` onto you laptop/friendly home machine.

1. `cd` (to make sure you are in your home directory)
2. `mkdir .ssh` (not needed if `~/.ssh` already exists)
3. `cd .ssh`
4. `scp _caneid_@thoreau.cs.miami.edu:~/.ssh/id_rsa_thoreau .` (and use your cane id password to authenticate; don't forget the dot at the end of the line)
5. `chmod go-rw id_rsa_thoreau`
6. `ssh -i id_rsa_thoreau _caneid_@thoreau.cs.miami.edu`

You should have logged in without any password prompts. If this did not happen, fix the situtation.

__Part 2__

There are three pieces of information in the ssh login line:

1. your cane id
2. the host name thoreau.ccs.miami.edu
3. the file name id_rsa_thoreau

Theses things can be written into the `~/.ssh/config` file and given a single name, thoreau, and then you can
log into triton with just the command `ssh thoreau`.

On your laptop/friendly machine, where you have the `~/.ssh/id_rsa_thoreau` file,

1. `cd` (to make sure you are in your home directory)
2. `cd .ssh`
3. `nano config`

Now put this into that file:

<pre>

Host thoreau
HostName thoreau.cs.miami.edu
User _caneid_
IdentityFile ~/.ssh/rsa_id_thoreau


</pre>

Exit nano with control-X, and confirm to save the changes with Y.

Now `ssh thoreau` and you should logon without a password. 

- You can also `scp thoreau:remote_file local_file` to copy
files from thoreau to your local machine, 
- or `scp local_file thoreau:remote_file` to copy files in the other direction.

Do not forget the : else it will think you are referring to a local file, scp is both cp and scp in one program.
