####################################################################
#                                                                  #
#  Copyright (c) 2015 California Institute of Technology.          #
#  Distributed under the MIT License.                              #
#  (See accompanying file LICENSE or copy at                       #
#  http://opensource.org/licenses/MIT)                             #
#                                                                  #
####################################################################
#                                                                  #
#   Coded by: Joseph Berleant (jberlean@caltech.edu)               #
#             Erik Winfree    (winfree@caltech.edu)                #
#             Chris Thachuk   (thachuk@caltech.edu)                #
#                                                                  #
#   Contributed code by: Justin Bois (bois@caltech.edu)            #
#                                                                  #
#                                                                  #
####################################################################

#
# The following functions are currently wrapped:
#  pfunc
#  pairs
#  mfe
#  subopt
#  count
#  energy
#  prob
#  complexdefect
#  sample
#
# The following functions may be wrapped in a future release:
#  complexes
#  concentrations
#  design
#  distributions

import math
import subprocess as sub
import os

os.environ["NUPACKHOME"] = "/data/ouga04b/ag_gagneur/home/karollus/nupack3.0.6"

def dGadjust(T,N):
    """Adjust NUPACK's native free energy (with reference to mole fraction units) to be appropriate for molar
    units, assuming N strands in the complex."""
    R=0.0019872041 # Boltzmann's constant in kcal/mol/K
    water=55.14    # molar concentration of water at 37 C, ignore temperature dependence, which is about 5%
    K=T+273.15     # Kelvin
    adjust = R*K*math.log(water) # converts from NUPACK mole fraction units to molar units, per association
    return adjust*(N-1)

def get_nupack_exec_path(exec_name):
    """ If the NUPACKHOME environment variable is set, use that as the directory
    of the NUPACK executables. Otherwise, have Python search the PATH directly. """
    if 'NUPACKHOME' in os.environ:
        return os.environ['NUPACKHOME'] + '/bin/' + exec_name
    else:
        return exec_name

def setup_args(**kargs):
    """ Returns the list of tokens specifying the command to be run in the pipe. """
    args = [get_nupack_exec_path(kargs['exec_name']),
          '-material', kargs['material'],   '-sodium', kargs['sodium'],
          '-magnesium', kargs['magnesium'], '-dangles', kargs['dangles'], '-T', kargs['T']]
    if kargs['multi']: args += ['-multi']
    if kargs['pseudo']: args += ['-pseudo']
    return args

def setup_cmd_input(multi, sequences, ordering, structure = ''):
    """ Returns the command-line input string to be given to NUPACK. """
    if not multi:
        cmd_input = '+'.join(sequences) + '\n' + structure
    else:
        n_seqs = len(sequences)
        if ordering == None:
             seq_order = ' '.join([str(i) for i in range(1, n_seqs+1)])
        else:
              seq_order = ' '.join([str(i) for i in ordering])
        cmd_input = str(n_seqs) + '\n' + ('\n'.join(sequences)) + '\n' + seq_order + '\n' + structure
    return cmd_input.strip()


def setup_nupack_input(**kargs):
    """ Returns the list of tokens specifying the command to be run in the pipe, and
    the command-line input to be given to NUPACK.
    Note that individual functions below may modify args or cmd_input depending on their
    specific usage specification. """
    # Set up terms of command-line executable call
    args = setup_args(**kargs)

    # Set up command-line input to NUPACK
    cmd_input = setup_cmd_input(kargs['multi'], kargs['sequences'], kargs['ordering'],
                                  kargs.get('structure', ''))

    return (args, cmd_input)

def call_with_file(args, cmd_input, outsuffix):
    """ Performs a NUPACK call, returning the lines of the output in a temporary
    output file. The output file is assumed to have the suffix 'outsuffix'.
  outsuffix includes the period (.) delimiter.
    Ex:
      call_with_file(args, input, '.sample')
  """

    import tempfile

    ## Preliminaries
    # Set up temporary output file
    outfile = tempfile.NamedTemporaryFile(delete=False, suffix=outsuffix)
    outprefix = outfile.name[:-len(outsuffix)]

    # Close the output file so sample can open/write to it.
    # Will reopen it later to get the output.
    outfile.close()

    ## Perform executable call, ignoring pipe output
    args = [str(s) for s in args] # all argument elements must be strings
    cmd_input = outprefix + '\n' + cmd_input # prepend the output file prefix to the input for NUPACK
    p = sub.Popen(args, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.STDOUT, encoding='utf8')
    p.communicate(cmd_input)

    ## Process and return output
    # Read output file and clean it up
    # Note that it was created by us, so it won't be cleaned up automatically
    out = open(outfile.name, "rt")
    output_lines = out.readlines()
    out.close()
    os.remove(outfile.name)
    return output_lines

def mfe(sequences, ordering = None, material = 'rna',
        dangles = 'some', T = 37, multi = True, pseudo = False,
        sodium = 1.0, magnesium = 0.0, degenerate = False):
    """Calls NUPACK's mfe executable on a complex consisting of the strands in sequences.
         Returns the minimum free energy structure, or multiple mfe structures if the degenerate
         option is specified
           sequences is a list of the strand sequences
           degenerate is a boolean specifying whether to include degenerate mfe structures
           See NUPACK User Manual for information on other arguments.
    """

    ## Set up command-line arguments and input
    args, cmd_input = \
        setup_nupack_input(exec_name = 'mfe', sequences = sequences, ordering = ordering,
                           material = material, sodium = sodium, magnesium = magnesium,
                           dangles = dangles, T = T, multi = multi, pseudo = pseudo)
    if degenerate: args += ['-degenerate']

    ## Perform call
    output = call_with_file(args, cmd_input, '.mfe')

    ## Parse and return output
    structs = []
    for i, l in enumerate(output):
        if l[0] == '.' or l[0] == '(':
            s = l.strip()
            e = output[i-1].strip()
            structs.append((s,e))

    return structs
