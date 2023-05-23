### SSH Public Key

Your Cane ID is your name you received at the Cane ID website. To manage or reset your password, visit https://caneidhelp.miami.edu/caneid/.

We refer to the password you used to create this account, as the Cane ID password. You have recieved other passwords for other machines,
such as the lab machines and for Pegasus. 

To help alleiviate the burden of passwords and other troublesome identifiers, as well as to enhance security, public key authentication is preferred
for ssh. It requires you create a public/private key pair using the command line program `ssh-keygen`. You can use it without parameters.
It will prompt for the name of the file to create. The default for an RSA key is `id_rsa`. 

The program will create two files: `id_rsa` and `id_rsa.pub`. The first contains the private key and <u>must</u> be kept secret.
The .pub file is the public key and you can share that freely. When you want to log into machine X with the `id_rsa` key you copy the 
__public__ key into .ssh/authorized_keys of host X. Then when you try to log into X, a math dance occurs were X learns you posses
the private key, with divulging the private key, and then allows the log in.

<pre>
  CLIENT                     SERVER
  chmod go-r .ssh/id_rsa     cat id_rsa.pub >> .ssh/authorized_keys 
  .ssh/id_rsa                .ssh/authorized_keys
</pre>

#### Example Johnston

##### On johnston

1. ssh to johnston.cs.miami.edu using your userid and password
2. `mkdir .ssh` # not needed if `~/.ssh` already exists
3. `cd .ssh`
4. `ssh-keygen -b 4096` 
5. `cat id_rsa >> authorized_keys`
6. logout

##### On your machine

1. `cd` (to make sure you are in your home directory)
2. `mkdir .ssh` # not needed if `~/.ssh` already exists
3. `cd .ssh`
4. use `scp` to copy `.ssh/id_rsa` from johnston to your machine
5. thatis: `scp -username-@johnston.cs.miami.edu:~/.ssh/id_rsa .` # will prompt for your password
6. `chmod go-rw id_rsa` # ssh will not work if keys are not protected. it's nanny-ware
7. now log in using the `-i` option to select the private key, 
8. `ssh -i id_rsa -username-@johnston.cs.miami.edu`

You should have logged in without any password prompts.

#### The config file

There are three pieces of information in the ssh login line:

1. your username
2. the host name johnston.miami.edu
3. the file name id_rsa

Theses things can be written into the `~/.ssh/config` file and given a single tag, say `johnston`, and 
then the ssh becomes simply `ssh johnston`. Ssh alwas looks in `~/.ssh/config` for options and adds
them automatically to your login attempt

Here is what goes into the config file,

<pre>
Host johnston
HostName johnston.cs.miami.edu
User _username_
IdentityFile ~/.ssh/rsa_id
</pre>

#### Ssh can copy files too!

The `scp` command uses the ssh protocol to copy files. With the config set up this is even easier, 

- `scp johnston:remote_file local_file`
- `scp local_file johnston:remote_file`

Do not forget the : else it will think you are referring to a local file.

### ProxyJump

The machine thoreau at present is not open to the internet. This is done to avoid opportunities for hackers. 
You log into johnston, which we have open to the internet, and then have ssh to direct you onwards to thoreau.
This way, from you point of view it is as if you are logging in directly to thoreau.

This is done with two entries in your conf file.

Host thoreau.via.johnston
User burt
Hostname 172.19.0.26
IdentityFile ~/.ssh/id_rsa_thoreau
ProxyJump armistead

Host johnston
User burt
Hostname johnston.cs.miami.edu
IdentityFile ~/.ssh/id_rsa_johnston

1. The private key for johnston is id_rsa_johnston, and is on your laptop. 
2. The matching public key is in .ssh/authorized_keys on johnston.
3. The private key for johnston is id_rsa_thoreau, and is on your laptop. 
4. The matching public key is in .ssh/authorized_keys on thoreau.
5. your ssh thoreau.via.johston

<pre>
CLIENT                   JOHSTON                                            THOREAU
id_rsa_johnston          cat id_rsa_johnston >> .ssh/authorized_keys        cat id_rsa_thoreau >> .ssh/authorized_keys
id_rsa_thoreau
</pre>
