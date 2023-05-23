## Public Key Cryptography


Public key cryptography was announced to the civilian world in 1976 in the paper "New Directions in Cryptography" by Whitfield Diffie and Martin E. Hellman. This was followed on 1977 by the paper
"A Method for Obtaining Digital Signatures and Public-Key Cryptosystems" by Ron Rivest, Adi Shamir, and Leonard Adleman, which proposed the
first known public key cryptosystem. I say "known" and "civilian" because it later became known that Clifford Cocks, a mathematician working for GCHQ had the idea in 1973, but it remained classified.

Ordinary crytosystems use a single key for encryption and decryption, and the idea is the two parties somehow share this key in a 
secure manner. Public key has two keys, one for encryption and the other for decryption. The decryption key is kept secret but
the encryption key is made public. Therefore what is shared is the public key, and it can be done over a public channel.

## SSH

Ssh is a terminal communication application that uses the SSL protocol to assure confidentially by using public key cryptography
to establish an encrypted channel between your lap and the server 

The server has a private key, public key pair.
- When contacted by the client, it will handover its public key. 
- The client chooses a random key, encrypts it using the public key and sends it to the server.
- The server decrypts it. 
- Sharing a random key, the client and server use it to encrypt their further communication.
- Inside this encrypted channel, an standard username/password login is performed.

More details can be found in my [ssl tutorial](https://www.cs.miami.edu/home/burt/learning/Csc424.162/workbook/ssl-tutorial.html).

### Server authentication

Because you are giving your password over to the server, it is desirable that you trust the server. However, this is nothing 
in this protocol that assures you that the server you believe you are connecting to is the server you are connecting to. 
It would be better if you already had the public key, from a reliable source, rather than just accepting the public key on 
faith, given by the server. 

This is hard problem to solve, and mostly ssh solves it by a method called _key continuity_. What this mean is that when the 
server presents you its public key, you are warned and asked to accept the risk that the public key is in authentic. If you 
accept it, the key is memorized, and future connections are compared against the key. If they server key changes, and it is
a legitimate change, you will have to ask your machine to forget the old public key, and accept a new one.

### User authentication

So far, we have possibly authenticated the server, constructed an encryption channel, and passed our password over that channel.
But we can do better. We can also use public key cryptography to eleminate password authentication, and replace it with public
key (user) authentication.

You will create a public key, private key pair. You will share the public key with the server, and keep the private key private.
The channel established as before, the server will challenge you on your knowledge of the private key by encrypting something
withe public key, and seeing if you can decrypt it. 

This is better in a lot of ways. For us, the most obvious advantage is no more typing of passwords.

## SSH Step by Step

You will create a public key/private key pair using the program `ssh-keygen`.
The program will create two files. The first contains the private key and <u>must</u> be kept secret.
The second file as the same name but with .pub at the end. This is the public key and you will share that with 
the machine you want to log into.

The hidden directory `.ssh` contains keys and an `authorized_keys` file. The `authorized_keys` file is full of
the .pub keys of legitimate users. If a user has a matching private key for any of those public keys, the login succeeds.

The private keys by convention are stores in the `.ssh` directory, and must have read permission _only_ by user. The ssh
program will not proceed if this security requirement is not met.

<pre>
  CLIENT                     SERVER
  chmod go-r .ssh/id_rsa     cat id_rsa.pub >> .ssh/authorized_keys 
  .ssh/id_rsa                .ssh/authorized_keys
</pre>

We have johnston.cs.miami.edu open for ssh to the internet. You can login using your username/password. Here
are the details of how to log in with a public key.

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

## The config file

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
IdentityFile ~/.ssh/id_rsa
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
