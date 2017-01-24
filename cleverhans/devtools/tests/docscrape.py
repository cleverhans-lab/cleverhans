"""Extract reference documentation from the NumPy source tree.

"""

from __future__ import print_function

import inspect
from nose.plugins.skip import SkipTest
import re
import sys

from theano.compat import six


class Reader(object):

    """A line-based string reader.

    """

    def __init__(self, data):
        """
        Parameters
        ----------
        data : str
           String with lines separated by '\n'.

        """
        if isinstance(data, list):
            self._str = data
        else:
            self._str = data.split('\n')  # store string as list of lines

        self.reset()

    def __getitem__(self, n):
        return self._str[n]

    def reset(self):
        self._l = 0  # current line nr

    def read(self):
        if not self.eof():
            out = self[self._l]
            self._l += 1
            return out
        else:
            return ''

    def seek_next_non_empty_line(self):
        for l in self[self._l:]:
            if l.strip():
                break
            else:
                self._l += 1

    def eof(self):
        return self._l >= len(self._str)

    def read_to_condition(self, condition_func):
        start = self._l
        for line in self[start:]:
            if condition_func(line):
                return self[start:self._l]
            self._l += 1
            if self.eof():
                return self[start:self._l + 1]
        return []

    def read_to_next_empty_line(self):
        self.seek_next_non_empty_line()

        def is_empty(line):
            return not line.strip()
        return self.read_to_condition(is_empty)

    def read_to_next_unindented_line(self):
        def is_unindented(line):
            return (line.strip() and (len(line.lstrip()) == len(line)))
        return self.read_to_condition(is_unindented)

    def peek(self, n=0):
        if self._l + n < len(self._str):
            return self[self._l + n]
        else:
            return ''

    def is_empty(self):
        return not ''.join(self._str).strip()

    def __iter__(self):
        for line in self._str:
            yield line


class NumpyDocString(object):

    def __init__(self, docstring, name=None):
        if name:
            self.name = name
        docstring = docstring.split('\n')

        # De-indent paragraph
        try:
            indent = min(len(s) - len(s.lstrip()) for s in docstring
                         if s.strip())
        except ValueError:
            indent = 0

        for n, line in enumerate(docstring):
            docstring[n] = docstring[n][indent:]

        self._doc = Reader(docstring)
        self._parsed_data = {
            'Signature': '',
            'Summary': '',
            'Extended Summary': [],
            'Parameters': [],
            'Other Parameters': [],
            'Returns': [],
            'Raises': [],
            'Warns': [],
            'See Also': [],
            'Notes': [],
            'References': '',
            'Examples': '',
            'index': {},
            'Attributes': [],
            'Methods': [],
        }
        self.section_order = []

        self._parse()

    def __getitem__(self, key):
        return self._parsed_data[key]

    def __setitem__(self, key, val):
        if key not in self._parsed_data:
            raise ValueError("Unknown section %s" % key)
        else:
            self._parsed_data[key] = val

    def _is_at_section(self):
        self._doc.seek_next_non_empty_line()

        if self._doc.eof():
            return False

        l1 = self._doc.peek().strip()  # e.g. Parameters

        if l1.startswith('.. index::'):
            return True

        l2 = self._doc.peek(1).strip()  # ----------
        return (len(l1) == len(l2) and l2 == '-' * len(l1))

    def _strip(self, doc):
        i = 0
        j = 0
        for i, line in enumerate(doc):
            if line.strip():
                break

        for j, line in enumerate(doc[::-1]):
            if line.strip():
                break

        return doc[i:len(doc) - j]

    def _read_to_next_section(self):
        section = self._doc.read_to_next_empty_line()

        while not self._is_at_section() and not self._doc.eof():
            if not self._doc.peek(-1).strip():  # previous line was empty
                section += ['']

            section += self._doc.read_to_next_empty_line()

        return section

    def _read_sections(self):
        while not self._doc.eof():
            data = self._read_to_next_section()
            name = data[0].strip()

            if name.startswith('..'):  # index section
                yield name, data[1:]
            elif len(data) < 2:
                yield StopIteration
            else:
                yield name, self._strip(data[2:])

    def _parse_param_list(self, content):
        r = Reader(content)
        params = []
        while not r.eof():
            header = r.read().strip()
            if ' : ' in header:
                arg_name, arg_type = header.split(' : ')[:2]
            else:
                arg_name, arg_type = header, ''

            desc = r.read_to_next_unindented_line()
            for n, line in enumerate(desc):
                desc[n] = line.strip()
            desc = desc  # '\n'.join(desc)

            params.append((arg_name, arg_type, desc))

        return params

    def _parse_see_also(self, content):
        """
        func_name : Descriptive text
            continued text
        another_func_name : Descriptive text
        func_name1, func_name2, func_name3

        """
        functions = []
        current_func = None
        rest = []
        for line in content:
            if not line.strip():
                continue
            if ':' in line:
                if current_func:
                    functions.append((current_func, rest))
                r = line.split(':', 1)
                current_func = r[0].strip()
                r[1] = r[1].strip()
                if r[1]:
                    rest = [r[1]]
                else:
                    rest = []
            elif not line.startswith(' '):
                if current_func:
                    functions.append((current_func, rest))
                    current_func = None
                    rest = []
                if ',' in line:
                    for func in line.split(','):
                        func = func.strip()
                        if func:
                            functions.append((func, []))
                elif line.strip():
                    current_func = line.strip()
            elif current_func is not None:
                rest.append(line.strip())
        if current_func:
            functions.append((current_func, rest))
        return functions

    def _parse_index(self, section, content):
        """
        .. index: default
           :refguide: something, else, and more

        """
        def strip_each_in(lst):
            return [s.strip() for s in lst]

        out = {}
        section = section.split('::')
        if len(section) > 1:
            out['default'] = strip_each_in(section[1].split(','))[0]
        for line in content:
            line = line.split(':')
            if len(line) > 2:
                out[line[1]] = strip_each_in(line[2].split(','))
        return out

    def _parse_summary(self):
        """Grab signature (if given) and summary"""
        summary = self._doc.read_to_next_empty_line()
        summary_str = "\n".join([s.strip() for s in summary])
        if re.compile('^([\w. ]+=)?[\w\.]+\(.*\)$').match(summary_str):
            self['Signature'] = summary_str
            if not self._is_at_section():
                self['Summary'] = self._doc.read_to_next_empty_line()
        elif re.compile('^[\w]+\n[-]+').match(summary_str):
            self['Summary'] = ''
            self._doc.reset()
        else:
            self['Summary'] = summary

        if not self._is_at_section():
            self['Extended Summary'] = self._read_to_next_section()

    def _parse(self):
        self._doc.reset()
        self._parse_summary()
        for (section, content) in self._read_sections():
            if not section.startswith('..'):
                section = ' '.join([s.capitalize()
                                    for s in section.split(' ')])
            if section in ('Parameters', 'Other Parameters', 'Returns',
                           'Raises', 'Warns', 'Attributes', 'Methods'):
                self[section] = self._parse_param_list(content)
                self.section_order.append(section)
            elif section.startswith('.. index::'):
                self['index'] = self._parse_index(section, content)
                self.section_order.append('index')
            elif section.lower() == 'see also':
                self['See Also'] = self._parse_see_also(content)
                self.section_order.append('See Also')
            else:
                self[section] = content
                self.section_order.append(section)

    # string conversion routines

    def _str_header(self, name, symbol='-'):
        return [name, len(name) * symbol]

    def _str_indent(self, doc, indent=4):
        out = []
        for line in doc:
            out += [' ' * indent + line]
        return out

    def _str_signature(self):
        if not self['Signature']:
            return []
        return ["*%s*" % self['Signature'].replace('*', '\*')] + ['']

    def _str_summary(self):
        return self['Summary'] + ['']

    def _str_extended_summary(self):
        return self['Extended Summary'] + ['']

    def _str_param_list(self, name):
        out = []
        if self[name]:
            out += self._str_header(name)
            for param, param_type, desc in self[name]:
                out += ['%s : %s' % (param, param_type)]
                out += self._str_indent(desc)
            out += ['']
        return out

    def _str_section(self, name):
        out = []
        if self[name]:
            out += self._str_header(name)
            out += self[name]
            out += ['']
        return out

    def _str_see_also(self):
        if not self['See Also']:
            return []
        out = []
        out += self._str_header("See Also")
        last_had_desc = True
        for func, desc in self['See Also']:
            if desc or last_had_desc:
                out += ['']
                out += ["`%s`_" % func]
            else:
                out[-1] += ", `%s`_" % func
            if desc:
                out += self._str_indent(desc)
                last_had_desc = True
            else:
                last_had_desc = False
        out += ['']
        return out

    def _str_index(self):
        idx = self['index']
        out = []
        out += ['.. index:: %s' % idx.get('default', '')]
        for section, references in six.iteritems(idx):
            if section == 'default':
                continue
            out += ['   :%s: %s' % (section, ', '.join(references))]
        return out

    def __str__(self):
        out = []
        out += self._str_signature()
        out += self._str_summary()
        out += self._str_extended_summary()
        for param_list in ('Parameters', 'Other Parameters',
                           'Returns', 'Raises', 'Warns'):
            out += self._str_param_list(param_list)
        out += self._str_see_also()
        for s in ('Notes', 'References', 'Examples'):
            out += self._str_section(s)
        out += self._str_index()
        return '\n'.join(out)

    # --

    def get_errors(self, check_order=True):
        errors = []
        self._doc.reset()
        for j, line in enumerate(self._doc):
            if len(line) > 75:
                if hasattr(self, 'name'):
                    errors.append("%s: Line %d exceeds 75 chars"
                                  ": \"%s\"..." % (self.name, j + 1,
                                                   line[:30]))
                else:
                    errors.append("Line %d exceeds 75 chars"
                                  ": \"%s\"..." % (j + 1, line[:30]))

        if check_order:
            canonical_order = ['Signature', 'Summary', 'Extended Summary',
                               'Attributes', 'Methods', 'Parameters',
                               'Other Parameters', 'Returns', 'Raises',
                               'Warns',
                               'See Also', 'Notes', 'References', 'Examples',
                               'index']

            canonical_order_copy = list(canonical_order)

            for s in self.section_order:
                while canonical_order_copy and s != canonical_order_copy[0]:
                    canonical_order_copy.pop(0)
                    if not canonical_order_copy:
                        errors.append(
                            "Sections in wrong order (starting at %s). The"
                            " right order is %s" % (s, canonical_order))

        return errors


def indent(str, indent=4):
    indent_str = ' ' * indent
    if str is None:
        return indent_str
    lines = str.split('\n')
    return '\n'.join(indent_str + l for l in lines)


class NumpyFunctionDocString(NumpyDocString):

    def __init__(self, docstring, function):
        super(NumpyFunctionDocString, self).__init__(docstring)
        args, varargs, keywords, defaults = inspect.getargspec(function)
        if (args and args != ['self']) or varargs or keywords or defaults:
            self.has_parameters = True
        else:
            self.has_parameters = False

    def _parse(self):
        self._parsed_data = {
            'Signature': '',
            'Summary': '',
            'Extended Summary': [],
            'Parameters': [],
            'Other Parameters': [],
            'Returns': [],
            'Raises': [],
            'Warns': [],
            'See Also': [],
            'Notes': [],
            'References': '',
            'Examples': '',
            'index': {},
        }
        return NumpyDocString._parse(self)

    def get_errors(self):
        errors = NumpyDocString.get_errors(self)

        if not self['Signature']:
            # This check is currently too restrictive.
            # Disabling it for now.
            # errors.append("No function signature")
            pass

        if not self['Summary']:
            errors.append("No function summary line")

        if len(" ".join(self['Summary'])) > 3 * 80:
            errors.append("Brief function summary is longer than 3 lines")

        if not self['Parameters'] and self.has_parameters:
            errors.append("No Parameters section")

        return errors


class NumpyClassDocString(NumpyDocString):

    def __init__(self, docstring, class_name, class_object):
        super(NumpyClassDocString, self).__init__(docstring)
        self.class_name = class_name
        methods = dict((name, func) for name, func
                       in inspect.getmembers(class_object))

        self.has_parameters = False
        if '__init__' in methods:
            # verify if __init__ is a Python function. If it isn't
            # (e.g. the function is implemented in C), getargspec will fail
            if not inspect.ismethod(methods['__init__']):
                return
            args, varargs, keywords, defaults = inspect.getargspec(
                methods['__init__'])
            if (args and args != ['self']) or varargs or keywords or defaults:
                self.has_parameters = True

    def _parse(self):
        self._parsed_data = {
            'Signature': '',
            'Summary': '',
            'Extended Summary': [],
            'Parameters': [],
            'Other Parameters': [],
            'Raises': [],
            'Warns': [],
            'See Also': [],
            'Notes': [],
            'References': '',
            'Examples': '',
            'index': {},
            'Attributes': [],
            'Methods': [],
        }
        return NumpyDocString._parse(self)

    def __str__(self):
        out = []
        out += self._str_signature()
        out += self._str_summary()
        out += self._str_extended_summary()
        for param_list in ('Attributes', 'Methods', 'Parameters', 'Raises',
                           'Warns'):
            out += self._str_param_list(param_list)
        out += self._str_see_also()
        for s in ('Notes', 'References', 'Examples'):
            out += self._str_section(s)
        out += self._str_index()
        return '\n'.join(out)

    def get_errors(self):
        errors = NumpyDocString.get_errors(self)
        if not self['Parameters'] and self.has_parameters:
            errors.append("%s class has no Parameters section"
                          % self.class_name)
        return errors


class NumpyModuleDocString(NumpyDocString):

    """
    Module doc strings: no parsing is done.

    """

    def _parse(self):
        self.out = []

    def __str__(self):
        return "\n".join(self._doc._str)

    def get_errors(self):
        errors = NumpyDocString.get_errors(self, check_order=False)
        return errors


def header(text, style='-'):
    return text + '\n' + style * len(text) + '\n'


class SphinxDocString(NumpyDocString):
    # string conversion routines

    def _str_header(self, name, symbol='`'):
        return ['**' + name + '**'] + [symbol * (len(name) + 4)]

    def _str_indent(self, doc, indent=4):
        out = []
        for line in doc:
            out += [' ' * indent + line]
        return out

    def _str_signature(self):
        return ['``%s``' % self['Signature'].replace('*', '\*')] + ['']

    def _str_summary(self):
        return self['Summary'] + ['']

    def _str_extended_summary(self):
        return self['Extended Summary'] + ['']

    def _str_param_list(self, name):
        out = []
        if self[name]:
            out += self._str_header(name)
            out += ['']
            for param, param_type, desc in self[name]:
                out += self._str_indent(['**%s** : %s' % (param, param_type)])
                out += ['']
                out += self._str_indent(desc, 8)
                out += ['']
        return out

    def _str_section(self, name):
        out = []
        if self[name]:
            out += self._str_header(name)
            out += ['']
            content = self._str_indent(self[name])
            out += content
            out += ['']
        return out

    def _str_index(self):
        idx = self['index']
        out = []
        out += ['.. index:: %s' % idx.get('default', '')]
        for section, references in six.iteritems(idx):
            if section == 'default':
                continue
            out += ['   :%s: %s' % (section, ', '.join(references))]
        return out

    def __str__(self, indent=0):
        out = []
        out += self._str_summary()
        out += self._str_extended_summary()
        for param_list in ('Parameters', 'Returns', 'Raises', 'Warns'):
            out += self._str_param_list(param_list)
        for s in ('Notes', 'References', 'Examples'):
            out += self._str_section(s)
        #        out += self._str_index()
        out = self._str_indent(out, indent)
        return '\n'.join(out)


class FunctionDoc(object):

    def __init__(self, func):
        self._f = func

    def __str__(self):
        out = ''
        doclines = inspect.getdoc(self._f) or ''
        try:
            doc = SphinxDocString(doclines)
        except Exception as e:
            print('*' * 78)
            print("ERROR: '%s' while parsing `%s`" % (e, self._f))
            print('*' * 78)
            # print "Docstring follows:"
            # print doclines
            # print '='*78
            return out

        if doc['Signature']:
            out += '%s\n' % header('**%s**' %
                                   doc['Signature'].replace('*', '\*'), '-')
        else:
            try:
                # try to read signature
                argspec = inspect.getargspec(self._f)
                argspec = inspect.formatargspec(*argspec)
                argspec = argspec.replace('*', '\*')
                out += header('%s%s' % (self._f.__name__, argspec), '-')
            except TypeError as e:
                out += '%s\n' % header('**%s()**' % self._f.__name__, '-')

        out += str(doc)
        return out


class ClassDoc(object):

    def __init__(self, cls, modulename=''):
        if not inspect.isclass(cls):
            raise ValueError("Initialise using an object")
        self._cls = cls

        if modulename and not modulename.endswith('.'):
            modulename += '.'
        self._mod = modulename
        self._name = cls.__name__

    @property
    def methods(self):
        return [name for name, func in inspect.getmembers(self._cls)
                if not name.startswith('_') and callable(func)]

    def __str__(self):
        out = ''

        def replace_header(match):
            return '"' * (match.end() - match.start())

        for m in self.methods:
            print("Parsing `%s`" % m)
            out += str(FunctionDoc(getattr(self._cls, m))) + '\n\n'
            out += '.. index::\n   single: %s; %s\n\n' % (self._name, m)

        return out


def handle_function(val, name):
    func_errors = []
    docstring = inspect.getdoc(val)
    if docstring is None:
        func_errors.append((name, '**missing** function-level docstring'))
    else:
        func_errors = [
            (name, e) for e in
            NumpyFunctionDocString(docstring, val).get_errors()
        ]
    return func_errors


def handle_module(val, name):
    module_errors = []
    docstring = val
    if docstring is None:
        module_errors.append((name, '**missing** module-level docstring'))
    else:
        module_errors = [
            (name, e) for e in NumpyModuleDocString(docstring).get_errors()
        ]
    return module_errors


def handle_method(method, method_name, class_name):
    method_errors = []

    # Skip out-of-library inherited methods
    module = inspect.getmodule(method)
    if module is not None:
        if not module.__name__.startswith('pylearn2'):
            return method_errors

    docstring = inspect.getdoc(method)
    if docstring is None:
        method_errors.append((class_name, method_name,
                              '**missing** method-level docstring'))
    else:
        method_errors = [
            (class_name, method_name, e) for e in
            NumpyFunctionDocString(docstring, method).get_errors()
        ]
    return method_errors


def handle_class(val, class_name):
    cls_errors = []
    docstring = inspect.getdoc(val)
    if docstring is None:
        cls_errors.append((class_name,
                           '**missing** class-level docstring'))
    else:
        cls_errors = [
            (e,) for e in
            NumpyClassDocString(docstring, class_name, val).get_errors()
        ]
        # Get public methods and parse their docstrings
        methods = dict(((name, func) for name, func in inspect.getmembers(val)
                        if not name.startswith('_') and callable(func) and
                        type(func) is not type))
        for m_name, method in six.iteritems(methods):
            # skip error check if the method was inherited
            # from a parent class (which means it wasn't
            # defined in this source file)
            if inspect.getmodule(method) is not None:
                continue
            cls_errors.extend(handle_method(method, m_name, class_name))
    return cls_errors


def docstring_errors(filename, global_dict=None):
    """
    Run a Python file, parse the docstrings of all the classes
    and functions it declares, and return them.

    Parameters
    ----------
    filename : str
        Filename of the module to run.

    global_dict : dict, optional
        Globals dictionary to pass along to `execfile()`.

    Returns
    -------
    all_errors : list
        Each entry of the list is a tuple, of length 2 or 3, with
        format either

        (func_or_class_name, docstring_error_description)
        or
        (class_name, method_name, docstring_error_description)
    """
    if global_dict is None:
        global_dict = {}
    if '__file__' not in global_dict:
        global_dict['__file__'] = filename
    if '__doc__' not in global_dict:
        global_dict['__doc__'] = None
    try:
        with open(filename) as f:
            code = compile(f.read(), filename, 'exec')
            exec(code, global_dict)
    except SystemExit:
        pass
    except SkipTest:
        raise AssertionError("Couldn't verify format of " + filename +
                             "due to SkipTest")
    all_errors = []
    for key, val in six.iteritems(global_dict):
        if not key.startswith('_'):
            module_name = ""
            if hasattr(inspect.getmodule(val), '__name__'):
                module_name = inspect.getmodule(val).__name__
            if (inspect.isfunction(val) or inspect.isclass(val)) and\
                    (inspect.getmodule(val) is None
                     or module_name == '__builtin__'):
                if inspect.isfunction(val):
                    all_errors.extend(handle_function(val, key))
                elif inspect.isclass(val):
                    all_errors.extend(handle_class(val, key))
        elif key == '__doc__':
            all_errors.extend(handle_module(val, key))
    if all_errors:
        all_errors.insert(0, ("%s:" % filename,))
    return all_errors


if __name__ == "__main__":
    all_errors = docstring_errors(sys.argv[1])
    if len(all_errors) > 0:
        print("*" * 30, "docstring errors", "*" * 30)
        for line in all_errors:
            print(':'.join(line))
    sys.exit(int(len(all_errors) > 0))
