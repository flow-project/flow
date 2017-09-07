setting up cistar
*****************************

To get cistar\_dev running, you need three things: cistar\_dev (or
CISTAR), SUMO, and rllab. Once each component is installed successfully,
you might get some missing module bugs from python. Just install the
missing module using your OS-specific package manager / installation
tool. Follow the shell commands below to get started.

Installing CISTAR
=================

1. ``git clone https://github.com/cathywu/learning-traffic.git``
2. Make sure you have access to the repo!
3. Add the ``cistar_dev`` directory to your ``PYTHONPATH`` environment
   variable. Visit the following links for OS-specific guides on how to
   do edit environment variables.
4. For Windows 10 users:
   https://www.computerhope.com/issues/ch000549.htm
5. For OS X users:
   http://osxdaily.com/2015/07/28/set-enviornment-variables-mac-os-x/
6. For Linux users:
   https://www.tecmint.com/set-unset-environment-variables-in-linux/ #
   Installing SUMO

SUMO can be installed using a subversion checkout, which will provide a
nightly build of SUMO. Apache Subversion is an open source software
versioning and revision control system. The current released version
(``0.30.0``) does not contain some functions that will be needed in
future research, such as the ability to change
speed-mode/speed-limits/other run parameters mid run. So for now the
nightly version is required. To complete the nightly build installation,
follow these steps:

1.  Install subversion (if necessary).
2.  Install the GNU autotools:
    https://en.wikipedia.org/wiki/GNU\_Build\_System. Calls to these
    autotools are hidden in Makefile.cvs, so you may get missing module
    errors when running step 6. To solve this, simply install the
    missing module using your OS-specific package manager / installation
    tool during step 6. Repeat until step 6 executes successfully.
3.  *NOTE:* this step is covered in the OS X setup guide in step 5, but
    with slightly different instructions. If you are running OS X,
    please using the guide to install the required dependencies instead.
4.  ``svn checkout``
    ``[https://svn.code.sf.net/p/sumo/code/trunk/sumo](https://svn.code.sf.net/p/sumo/code/trunk/sumo)``
5.  ``cd sumo``
6.  If at this point you are running OS X, please refer to the following
    setup guide instead of the next steps:
    http://sumo.dlr.de/wiki/Installing/MacOS\_Build\_w\_Homebrew
7.  ``make -f Makefile.cvs``
8.  ``make install``
9.  Add the ``sumo/tools`` directory to your ``PYTHONPATH`` environment
    variable. See instructions in ‘Installing CISTAR’ for how to edit
    environment variables.
10. Set SUMO\_HOME environment variable (for local schema look-up):
    ``export SUMO_HOME=<path to SUMO>`` # Installing rllab
11. Download and install Anaconda 4.4.0 Python 3.6 version for your OS:
    https://www.continuum.io/downloads
12. We are using rllab-distributed. Once Cathy gives you access, you can
    clone this repository onto your local machine:
    ``git clone https://github.com/openai/rllab-distributed.git``
13. Switch branches to ‘sharedPolicyUpdate’, which contains the
    necessary changes to GymEnv: ``git checkout sharedPolicyUpdate``
14. Now follow the installation steps on the rllab documentation
    website:
    http://rllab.readthedocs.io/en/latest/user/installation.html
15. *NOTE:* During the ‘Express install’ portion of the instructions,
    you are told to run these commands to activate / deactivate the
    created virtualenv:

    ::

         source activate rllab3 
         source deactivate rllab3

    Since we are using rllab-distributed, the virtualenv name is now
    ‘rllab-distributed’. Run the following commands instead:

    ::

         source activate rllab-distributed
         source deactivate rllab-distributed

Test the installation
=====================

Running the following should result in the loading of the sumo GUI.
Click the run button and you should see unstable traffic form after a
few seconds, a la (Sugiyama et al, 2008).

::

    source activate rllab-distributed
    cd <path to learning-traffic>/cistar_dev/examples
    python sugiyama.py

This means that you have cistar properly configured with SUMO.

::

    cd <path to learning-traffic>/cistar_dev/examples
    python mixed-rl-single-lane.py

This means that you have cistar properly configured with both SUMO and
rllab. Congratulations, you now have cistar set up!

Common Bugs
===========

‘Departure Time’ in SUMO Bug
----------------------------

In order to avoid “Departure Time” bugs, follow the steps in
/cistar\_dev/docs/sumo-depart-time-issue.md *NOTE:* in some nightly
builds, the line number to fix is 1172 instead of 1200.

‘Fx.h No Such File or Directory’ Bug
------------------------------------

When running the ‘make’ command while installing SUMO, you may face the
following error. It seems as though this bug occurs when the fox toolkit
has not been installed.

::

    In file included from MSParkingArea.cpp:36:0:
    ../../src/utils/foxtools/MFXMutex.h:37:16: fatal error: fx.h: No such file or directory

To fix this, install the fox toolkit: http://www.fox-toolkit.org/ Then
run ‘./configure’ and ‘make’ again.

‘PI Not In Scope’ Bug
---------------------

When running the ‘make’ command while installing SUMO, you may face the
following error. It also seems as though this bug occurs when the fox
toolkit has not been installed.

::

    Error:
    make[4]: Entering directory '.../sumo/src/microsim/devices'
    g++ -DHAVE_CONFIG_H -I. -I../../../src  -I.../sumo/./src -I/usr/include/gdal  -I/usr/local/include -I/usr/include    -msse2 -mfpmath=sse -std=c++11 -O2 -DNDEBUG  -MT MSDevice_SSM.o -MD -MP -MF .deps/MSDevice_SSM.Tpo -c -o MSDevice_SSM.o MSDevice_SSM.cpp
    MSDevice_SSM.cpp: In member function ‘MSDevice_SSM::EncounterType MSDevice_SSM::classifyEncounter(const MSDevice_SSM::FoeInfo*, MSDevice_SSM::EncounterApproachInfo&) const’:
    MSDevice_SSM.cpp:1664:126: error: ‘PI’ was not declared in this scope
     ionLine.rotationAtOffset(0.) - foeConnectionLine.rotationAtOffset(0.), (2*PI));
                                                                               ^
    Makefile:423: recipe for target 'MSDevice_SSM.o' failed
    make[4]: *** [MSDevice_SSM.o] Error 1
    make[4]: Leaving directory 'path-to-sumo/sumo/src/microsim/devices'
    Makefile:557: recipe for target 'all-recursive' failed
    make[3]: *** [all-recursive] Error 1
    make[3]: Leaving directory 'path-to-sumo/sumo/src/microsim'
    Makefile:688: recipe for target 'all-recursive' failed
    make[2]: *** [all-recursive] Error 1
    make[2]: Leaving directory 'path-to-sumo/sumo/src'
    Makefile:529: recipe for target 'all' failed
    make[1]: *** [all] Error 2
    make[1]: Leaving directory 'path-to-sumo/sumo/src'
    Makefile:405: recipe for target 'all-recursive' failed
    make: *** [all-recursive] Error 1

To fix this, install the fox toolkit: http://www.fox-toolkit.org/ Then
run ‘./configure’ and ‘make’ again.

‘String Pattern on Bytes-like Object’ Bug
-----------------------------------------

This is another bug you might face when running ‘make’ while installing
SUMO.

::

    Making all in src
    make[1]: Entering directory 'path-to-sumo/sumo/src'
    ../tools/build/version.py path-to-sumo/sumo/src
    Traceback (most recent call last):
     File "../tools/build/version.py", line 129, in main
       svnRevision = int(re.search('Revision: (\d*)', svnInfo).group(1))
     File "path-to-anaconda/anaconda3/lib/python3.6/re.py", line 182, in search
       return _compile(pattern, flags).search(string)
    TypeError: cannot use a string pattern on a bytes-like object

    During handling of the above exception, another exception occurred:

    Traceback (most recent call last):
     File "../tools/build/version.py", line 136, in <module>
       main()
     File "../tools/build/version.py", line 131, in main
       svnRevision = parseRevision(svnFile)
     File "../tools/build/version.py", line 71, in parseRevision
       m = re.search('[!]svn[/]ver[/](\d*)[/]', l)
     File "path-to-anaconda/anaconda3/lib/python3.6/re.py", line 182, in search
       return _compile(pattern, flags).search(string)
    TypeError: cannot use a string pattern on a bytes-like object
    Makefile:964: recipe for target 'version.h' failed
    make[1]: *** [version.h] Error 1
    make[1]: Leaving directory 'path-to-sumo/sumo/src'
    Makefile:405: recipe for target 'all-recursive' failed
    make: *** [all-recursive] Error 1

You can fix this by opening ‘path-to-sumo/sumo/tools/build/version.py’
and changing line 129 by wrapping the ‘svnInfo’ object in a repr() call.

from:

::

    svnRevision = int(re.search('Revision: (\d*)', svnInfo).group(1))

to:

::

    svnRevision = int(re.search('Revision: (\d*)', repr(svnInfo)).group(1))

‘Libtool Version Mismatch Error’ Bug
------------------------------------

This may happen while running ‘make’ for SUMO on Linux.

::

    libtool: Version mismatch error.  This is libtool 2.4.2, but the
    libtool: definition of this LT_INIT comes from libtool 2.4.6.
    libtool: You should recreate aclocal.m4 with macros from libtool 2.4.2
    libtool: and run autoconf again.
    Makefile:459: recipe for target 'marouter' failed
    make[3]: *** [marouter] Error 63

Try the steps under the ‘Troubleshooting’ section of this guide:
http://sumo.dlr.de/wiki/Installing/Linux\_Build

Refer to the ‘Problems with aclocal.m4 and libtool’ subsection of
‘Troubleshooting’. If it still doesn’t work, consider uninstalling
libtool and running the ‘Troubleshooting’ commands again.

‘No Module Named X’ Bugs When Running Examples
----------------------------------------------

While running a CISTAR example, you may get the following ‘no module
named ’ errors. This is because packages specified in the anaconda
environment file haven’t been correctly installed. Just use
``pip install <package-name>`` to fix this.

rllab atari error
-----------------

::

    Command "/Users/kathyjang/anaconda/envs/rllab-distributed/bin/python -u -c "import setuptools, tokenize;__file__='/private/var/folders/nl/glkpv_992fn6mjkftm3y0ywr0000gn/T/pip-build-_tf5oeha/atari-py/setup.py';f=getattr(tokenize, 'open', open)(__file__);code=f.read().replace('\r\n', '\n');f.close();exec(compile(code, __file__, 'exec'))" install --record /var/folders/nl/glkpv_992fn6mjkftm3y0ywr0000gn/T/pip-0czktx0o-record/install-record.txt --single-version-externally-managed --compile" failed with error code 1 in /private/var/folders/nl/glkpv_992fn6mjkftm3y0ywr0000gn/T/pip-build-_tf5oeha/atari-py/

Just ignore this. We don’t use atari.

When running an rl experiment in cistar, I get theano errors concerning .theano/compiledir\_Darwin-15.5.0-x86\_64-i386-64bit-i386-3.5.2-64 (or similar)
-------------------------------------------------------------------------------------------------------------------------------------------------------

Make sure you have an appropriate theano version installed:
``0.9.0dev1.dev-adfe319ce6b781083d8dc3200fb4481b00853791``

Check this by doing:

::

    python -c "import theano; print(theano.__version__)"

If this is not the case, then do the following:

::

    pip install theano==0.8.2

If this continues to error, then try running the experiment in
``local_docker`` mode instead of ``local`` mode.

Missing Traci module
--------------------

Add /path/to/sumo/tools to your PYTHONPATH
