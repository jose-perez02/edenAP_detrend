"""
This module will contain functions and/or classes that facilitate directory and files management.
"""
import os
import subprocess


def walklevel(some_dir, level=100, hidden=False, dirs_filter=False):
    """
    Generator wrapper of os.walk() that walks only on a number of subfolder branches
    :param some_dir: directory to be walked
    :param level: number of subdirectories to look into. Default: 100 (for complete walk)
    :param hidden: If true it will recurse through hidden folders, otherwise it will filter them out
    :param dirs_filter: filter function (callable) that allows folder filtering
    :return: generator that yields tuple (dirpath, dirname, filenames)
    """
    # strip separator from end of string
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir), "Invalid directory string, try again"
    # number of starting dirs
    num_sep = some_dir.count(os.path.sep)
    for dirpath, dirnames, filenames in os.walk(some_dir):
        if not hidden:
            # avoid hidden folders/files
            dirnames[:] = [d for d in dirnames if not d[0] == '.']
            filenames = [f for f in filenames if not f[0] == '.']
        if dirs_filter:
            dirnames[:] = [d for d in dirnames if dirs_filter(d)]
        yield dirpath, dirnames, filenames
        # number of dirs in this one folder
        num_sep_this = dirpath.count(os.path.sep)
        # delete rest of dirs if limit has been reached
        if num_sep + level <= num_sep_this:
            del dirnames[:]


def new_destination(source, dest):
    """
    Simple utility for copy/mv functions to infer destination and avoid replacement of files.
    File will be renamed according to existing files.
    :param source: source file
    :param dest: destination, either folder or final filepath
    :return: fixed final filapath

    Example:
    > MyDirectory
        > MyFile

    >> new_destination('../MyFile', './MyDirectory')

    > MyDirectory
        > MyFile
        > MyFile(copy0)

    >> new_destination('../MyFile', './MyDirectory')

    > MyDirectory
        > MyFile
        > MyFile(copy0)
        > MyFile(copy1)
    """
    join = os.path.join
    if os.path.isdir(dest):
        filename = source.split(os.sep)[-1]
        dirpath = dest
    else:
        filename = dest.split(os.sep)[-1]
        dirpath = os.path.dirname(dest)
    isFile = os.path.isfile(join(dirpath, filename))
    count = 1
    while isFile:
        name, extension = '.'.join(filename.split('.')[:-1]), filename.split('.')[-1]
        newfilename = name + "(copy%d)" % count + '.' + extension
        print("File {} already exists in destination folder... Changing filename to {}".format(filename, newfilename))
        dest = join(dirpath, newfilename)
        isFile = os.path.isfile(dest)
        count += 1
    return dest


def copy(source, dest):
    """
    Wrapper function that uses unix system's copy function: `cp -n`
    :param source: source file
    :param dest: destination file/folder
    :return:
    """
    dest = new_destination(source, dest)
    proc = subprocess.Popen(['cp', '-n', source, dest], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return_code = proc.wait()
    if return_code == 1:
        print("Copy Function encountered Error. File somehow exists already.")
    return return_code


def mv(source, dest):
    """
    Wrapper function that uses unix system's move function: `mv -n`
    :param source: source file
    :param dest: destination file/folder
    :return:
    """
    dest = new_destination(source, dest)
    proc = subprocess.Popen(['mv', '-n', source, dest], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return_code = proc.wait()
    if return_code == 1:
        print("Move Function encountered Error. File somehow exists already.")
    return return_code


def shorten_path(path):
    """
    function to shorten a path to a file. For displaying purposes.
    :param path: path to file
    :return:
    """
    path_list = path.split(os.sep)
    if len(path_list) < 5:
        short_path = os.path.join("~", *path_list[:-1])
    else:
        short_path = os.path.join("~", *path_list[-5:-1])
    return short_path


def set_leaf(tree, branches, leaf):
    """ Set a terminal element to *leaf* within nested dictionaries.
    *branches* defines the path through dictionaries.

    Example:
    >>> t = {}
    >>> set_leaf(t, ['b1','b2','b3'], 'new_leaf')
    >>> print(t)
    {'b1': {'b2': {'b3': 'new_leaf'}}}
    """
    if len(branches) == 1:
        tree[branches[0]] = leaf
        return
    if not branches[0] in tree:
        tree[branches[0]] = {}
    set_leaf(tree[branches[0]], branches[1:], leaf)


def dir_tree(start_path, level=1000) -> dict:
    """
    Return directory structure as a multi-level dictionary. Every key is a folder, and files are
    dictionaries with single value of None.
    """
    tree = {}
    for root, dirs, files in walklevel(start_path, level=level):
        branches = [start_path]
        if root != start_path:
            branches.extend(os.path.relpath(root, start_path).split(os.sep))
        set_leaf(tree, branches, dict([(d, {}) for d in dirs] + [(f, None) for f in files]))
    return tree[start_path]


# function to validate directory paths. It creates a path if it doesn't already exist.
def validate_dirs(*paths):
    """
    Validate directories. Create directory tree if it doesn't exist. Any number of arguments (paths) are valid.
    """
    for folder_path in paths:
        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)
