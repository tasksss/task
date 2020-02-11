from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import time

from blessings import Terminal

from gpustat import __version__
#from .core import GPUStatCollection

# core.py
import json
import locale
import os.path
import platform
import sys
import time
from datetime import datetime

from six.moves import cStringIO as StringIO
import psutil
import pynvml as N
from blessings import Terminal

#import gpustat.util as util

NOT_SUPPORTED = 'Not Supported'
MB = 1024 * 1024


class GPUStat(object):

    def __init__(self, entry):
        if not isinstance(entry, dict):
            raise TypeError(
                'entry should be a dict, {} given'.format(type(entry))
            )
        self.entry = entry

    def __repr__(self):
        return self.print_to(StringIO()).getvalue()

    def keys(self):
        return self.entry.keys()

    def __getitem__(self, key):
        return self.entry[key]

    # to contnue

    def print_to(self, fp,
                 with_colors=True,  # deprecated arg
                 #show_cmd=False,
                 #show_full_cmd=False,
                 show_user=False,
                 show_pid=False,
                 show_power=None,
                 show_fan_speed=None,
                 gpuname_width=16,
                 term=Terminal(),
                 ):
        # color settings
        colors = {}

        def _conditional(cond_fn, true_value, false_value,
                         error_value=term.bold_black):
            try:
                return cond_fn() and true_value or false_value
            except Exception:
                return error_value

        colors['C0'] = term.normal
        colors['C1'] = term.cyan
        colors['CName'] = term.blue
        # colors['CTemp'] = _conditional(lambda: self.temperature < 50,
        #                                term.red, term.bold_red)
        # colors['FSpeed'] = _conditional(lambda: self.fan_speed < 30,
        #                                 term.cyan, term.bold_cyan)
        # colors['CMemU'] = term.bold_yellow
        # colors['CMemT'] = term.yellow
        # colors['CMemP'] = term.yellow
        # colors['CCPUMemU'] = term.yellow
        colors['CUser'] = term.bold_black  # gray
        # colors['CUtil'] = _conditional(lambda: self.utilization < 30,
        #                                term.green, term.bold_green)
        # colors['CCPUUtil'] = term.green
        # colors['CPowU'] = _conditional(
        #     lambda: float(self.power_draw) / self.power_limit < 0.4,
        #     term.magenta, term.bold_magenta
        # )
        # colors['CPowL'] = term.magenta
        # colors['CCmd'] = term.color(24)   # a bit dark

        if not with_colors:
            for k in list(colors.keys()):
                colors[k] = ''

        def _repr(v, none_value='??'):
            return none_value if v is None else v

        # build one-line display information
        # we want power use optional, but if deserves being grouped with
        # temperature and utilization
        reps = u"%(C1)s[{entry[index]}]%(C0)s " \
               "%(CName)s{entry[name]:{gpuname_width}}%(C0)s |"  # \
        # "%(CTemp)s{entry[temperature.gpu]:>3}°C%(C0)s, "

        # if show_fan_speed:
        #     reps += "%(FSpeed)s{entry[fan.speed]:>3} %%%(C0)s, "
        #
        # reps += "%(CUtil)s{entry[utilization.gpu]:>3} %%%(C0)s"

        # if show_power:
        #     reps += ",  %(CPowU)s{entry[power.draw]:>3}%(C0)s "
        #     if show_power is True or 'limit' in show_power:
        #         reps += "/ %(CPowL)s{entry[enforced.power.limit]:>3}%(C0)s "
        #         reps += "%(CPowL)sW%(C0)s"
        #     else:
        #         reps += "%(CPowU)sW%(C0)s"

        # reps += " | %(C1)s%(CMemU)s{entry[memory.used]:>5}%(C0)s " \
        #     "/ %(CMemT)s{entry[memory.total]:>5}%(C0)s MB"
        reps = (reps) % colors
        reps = reps.format(entry={k: _repr(v) for k, v in self.entry.items()},
                           gpuname_width=gpuname_width)
        reps += " |"

        def process_repr(p):
            r = ''
            if show_user: #not show_cmd or
                r += "{CUser}{}{C0}".format(
                    _repr(p['username'], '--'), **colors
                )
            # if show_cmd:
            #     if r:
            #         r += ':'
            #     r += "{C1}{}{C0}".format(
            #         _repr(p.get('command', p['pid']), '--'), **colors
            #     )

            if show_pid:
                r += ("/%s" % _repr(p['pid'], '--'))
            # r += '({CMemP}{}M{C0})'.format(
            #     _repr(p['gpu_memory_usage'], '?'), **colors
            # )
            return r

        def full_process_info(p):
            r = "{C0} ├─ {:>6} ".format(
                _repr(p['pid'], '--'), **colors
            )
            # r += "{C0}({CCPUUtil}{:4.0f}%{C0}, {CCPUMemU}{:>6}{C0})".format(
            #         _repr(p['cpu_percent'], '--'),
            #         util.bytes2human(_repr(p['cpu_memory_usage'], 0)), **colors
            #     )
            # full_command_pretty = util.prettify_commandline(
            #     p['full_command'], colors['C1'], colors['CCmd'])
            # r += "{C0}: {CCmd}{}{C0}".format(
            #     _repr(full_command_pretty, '?'),
            #     **colors
            # )
            return r

        processes = self.entry['processes']
        full_processes = []
        if processes is None:
            # None (not available)
            reps += ' ({})'.format(NOT_SUPPORTED)
        else:
            for p in processes:
                reps += ' ' + process_repr(p)
                # if show_full_cmd:
                #     full_processes.append('\n' + full_process_info(p))
        if full_processes: # show_full_cmd and
            full_processes[-1] = full_processes[-1].replace('├', '└', 1)
            reps += ''.join(full_processes)
        fp.write(reps)
        return fp

    def jsonify(self):
        o = dict(self.entry)
        if self.entry['processes'] is not None:
            o['processes'] = [{k: v for (k, v) in p.items() if k != 'gpu_uuid'}
                              for p in self.entry['processes']]
        else:
            o['processes'] = '({})'.format(NOT_SUPPORTED)
        return o


class GPUStatCollection(object):
    global_processes = {}

    def __init__(self, gpu_list, driver_version=None):
        self.gpus = gpu_list

        # attach additional system information
        self.hostname = platform.node()
        self.query_time = datetime.now()
        self.driver_version = driver_version

    @staticmethod
    def clean_processes():
        for pid in list(GPUStatCollection.global_processes.keys()):
            if not psutil.pid_exists(pid):
                del GPUStatCollection.global_processes[pid]

    @staticmethod
    def new_query():
        """Query the information of all the GPUs on local machine"""

        N.nvmlInit()

        def _decode(b):
            if isinstance(b, bytes):
                return b.decode()  # for python3, to unicode
            return b

        def get_gpu_info(handle):
            """Get one GPU information specified by nvml handle"""

            def get_process_info(nv_process):
                """Get the process information of specific pid"""
                process = {}
                if nv_process.pid not in GPUStatCollection.global_processes:
                    GPUStatCollection.global_processes[nv_process.pid] = \
                        psutil.Process(pid=nv_process.pid)
                ps_process = GPUStatCollection.global_processes[nv_process.pid]
                process['username'] = ps_process.username()

                # _cmdline = ps_process.cmdline()
                # if not _cmdline:

                #     process['command'] = '?'
                #     process['full_command'] = ['?']
                # else:
                #     process['command'] = os.path.basename(_cmdline[0])
                #     process['full_command'] = _cmdline

                # process['gpu_memory_usage'] = nv_process.usedGpuMemory // MB
                # process['cpu_percent'] = ps_process.cpu_percent()
                # process['cpu_memory_usage'] = \
                #     round((ps_process.memory_percent() / 100.0) *
                #           psutil.virtual_memory().total)
                process['pid'] = nv_process.pid
                return process

            name = _decode(N.nvmlDeviceGetName(handle))
            uuid = _decode(N.nvmlDeviceGetUUID(handle))

            # try:
            #     temperature = N.nvmlDeviceGetTemperature(
            #         handle, N.NVML_TEMPERATURE_GPU
            #     )
            # except N.NVMLError:
            #     temperature = None  # Not supported

            # try:
            #     fan_speed = N.nvmlDeviceGetFanSpeed(handle)
            # except N.NVMLError:
            #     fan_speed = None  # Not supported

            # try:
            #     memory = N.nvmlDeviceGetMemoryInfo(handle)  # in Bytes
            # except N.NVMLError:
            #     memory = None  # Not supported

            # try:
            #     utilization = N.nvmlDeviceGetUtilizationRates(handle)
            # except N.NVMLError:
            #     utilization = None  # Not supported

            # try:
            #     power = N.nvmlDeviceGetPowerUsage(handle)
            # except N.NVMLError:
            #     power = None

            # try:
            #     power_limit = N.nvmlDeviceGetEnforcedPowerLimit(handle)
            # except N.NVMLError:
            #     power_limit = None

            try:
                nv_comp_processes = \
                    N.nvmlDeviceGetComputeRunningProcesses(handle)
            except N.NVMLError:
                nv_comp_processes = None  # Not supported
            try:
                nv_graphics_processes = \
                    N.nvmlDeviceGetGraphicsRunningProcesses(handle)
            except N.NVMLError:
                nv_graphics_processes = None  # Not supported

            if nv_comp_processes is None and nv_graphics_processes is None:
                processes = None
            else:
                processes = []
                nv_comp_processes = nv_comp_processes or []
                nv_graphics_processes = nv_graphics_processes or []
                for nv_process in nv_comp_processes + nv_graphics_processes:
                    try:
                        process = get_process_info(nv_process)
                        processes.append(process)
                    except psutil.NoSuchProcess:
                        # TODO: add some reminder for NVML broken context
                        # e.g. nvidia-smi reset  or  reboot the system
                        pass

                # TODO: Do not block if full process info is not requested
                time.sleep(0.1)
                for process in processes:
                    pid = process['pid']
                    cache_process = GPUStatCollection.global_processes[pid]
                    # process['cpu_percent'] = cache_process.cpu_percent()

            index = N.nvmlDeviceGetIndex(handle)
            gpu_info = {
                'index': index,
                'uuid': uuid,
                'name': name,
                # 'temperature.gpu': temperature,
                # 'fan.speed': fan_speed,
                # 'utilization.gpu': utilization.gpu if utilization else None,
                # 'power.draw': power // 1000 if power is not None else None,
                # 'enforced.power.limit': power_limit // 1000
                # if power_limit is not None else None,
                # Convert bytes into MBytes
                # 'memory.used': memory.used // MB if memory else None,
                # 'memory.total': memory.total // MB if memory else None,
                'processes': processes,
            }
            GPUStatCollection.clean_processes()
            return gpu_info

        # 1. get the list of gpu and status
        gpu_list = []
        device_count = N.nvmlDeviceGetCount()

        for index in range(device_count):
            handle = N.nvmlDeviceGetHandleByIndex(index)
            gpu_info = get_gpu_info(handle)
            gpu_stat = GPUStat(gpu_info)
            gpu_list.append(gpu_stat)

        # 2. additional info (driver version, etc).
        try:
            driver_version = _decode(N.nvmlSystemGetDriverVersion())
        except N.NVMLError:
            driver_version = None  # N/A

        N.nvmlShutdown()
        return GPUStatCollection(gpu_list, driver_version=driver_version)

    def __len__(self):
        return len(self.gpus)

    def __iter__(self):
        return iter(self.gpus)

    def __getitem__(self, index):
        return self.gpus[index]

    def __repr__(self):
        s = 'GPUStatCollection(host=%s, [\n' % self.hostname
        s += '\n'.join('  ' + str(g) for g in self.gpus)
        s += '\n])'
        return s

    # --- Printing Functions ---

    def print_formatted(self, fp=sys.stdout, force_color=False, no_color=False,
                        # show_cmd=False, show_full_cmd=False,
                        show_user=False,
                        show_pid=False,
                        # show_power=None, show_fan_speed=None,
                        gpuname_width=16,
                        show_header=True,
                        eol_char=os.linesep,
                        ):
        # ANSI color configuration
        if force_color and no_color:
            raise ValueError("--color and --no_color can't"
                             " be used at the same time")

        if force_color:
            t_color = Terminal(kind='linux', force_styling=True)

            # workaround of issue #32 (watch doesn't recognize sgr0 characters)
            t_color.normal = u'\x1b[0;10m'
        elif no_color:
            t_color = Terminal(force_styling=None)
        else:
            t_color = Terminal()  # auto, depending on isatty

        # appearance settings
        entry_name_width = [len(g.entry['name']) for g in self]
        gpuname_width = max([gpuname_width or 0] + entry_name_width)

        # header
        if show_header:
            time_format = locale.nl_langinfo(locale.D_T_FMT)

            header_template = '{t.bold_white}{hostname:{width}}{t.normal}  '
            header_template += '{timestr}  '
            header_template += '{t.bold_black}{driver_version}{t.normal}'

            header_msg = header_template.format(
                hostname=self.hostname,
                width=gpuname_width + 3,  # len("[?]")
                timestr=self.query_time.strftime(time_format),
                driver_version=self.driver_version,
                t=t_color,
            )

            fp.write(header_msg.strip())
            fp.write(eol_char)

        # body
        for g in self:
            g.print_to(fp,
                       # show_cmd=show_cmd,
                       # show_full_cmd=show_full_cmd,
                       show_user=show_user,
                       show_pid=show_pid,
                       # show_power=show_power,
                       # show_fan_speed=show_fan_speed,
                       gpuname_width=gpuname_width,
                       term=t_color)
            fp.write(eol_char)

        fp.flush()

    def jsonify(self):
        return {
            'hostname': self.hostname,
            'query_time': self.query_time,
            "gpus": [g.jsonify() for g in self]
        }

    def print_json(self, fp=sys.stdout):
        def date_handler(obj):
            if hasattr(obj, 'isoformat'):
                return obj.isoformat()
            else:
                raise TypeError(type(obj))

        o = self.jsonify()
        json.dump(o, fp, indent=4, separators=(',', ': '),
                  default=date_handler)
        fp.write('\n')
        fp.flush()


def new_query():
    '''
    Obtain a new GPUStatCollection instance by querying nvidia-smi
    to get the list of GPUs and running process information.
    '''
    return GPUStatCollection.new_query()

#cli.py
def print_gpustat(json=False, debug=False, **kwargs):
    '''
    Display the GPU query results into standard output.
    '''
    try:
        gpu_stats = GPUStatCollection.new_query()
    except Exception as e:
        sys.stderr.write('Error on querying NVIDIA devices.'
                         ' Use --debug flag for details\n')
        if debug:
            try:
                import traceback
                traceback.print_exc(file=sys.stderr)
            except Exception:
                # NVMLError can't be processed by traceback:
                #   https://bugs.python.org/issue28603
                # as a workaround, simply re-throw the exception
                raise e
        sys.exit(1)

    if json:
        gpu_stats.print_json(sys.stdout)
    else:
        gpu_stats.print_formatted(sys.stdout, **kwargs)


def loop_gpustat(interval=1.0, **kwargs):
    term = Terminal()

    with term.fullscreen():
        while 1:
            try:
                query_start = time.time()
                with term.location(0, 0):
                    print_gpustat(eol_char=term.clear_eol + '\n', **kwargs)
                    print(term.clear_eos, end='')
                query_duration = time.time() - query_start
                sleep_duration = interval - query_duration
                if sleep_duration > 0:
                    time.sleep(sleep_duration)
            except KeyboardInterrupt:
                return 0


def main(*argv):
    if not argv:
        argv = list(sys.argv)

    # attach SIGPIPE handler to properly handle broken pipe
    import signal
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    # arguments to gpustat
    import argparse
    parser = argparse.ArgumentParser()

    parser_color = parser.add_mutually_exclusive_group()
    parser_color.add_argument('--force-color', '--color', action='store_true',
                              help='Force to output with colors')
    parser_color.add_argument('--no-color', action='store_true',
                              help='Suppress colored output')
    # parser.add_argument('-a', '--show-all', action='store_true',
    #                     help='Display all gpu properties above')

    # parser.add_argument('-c', '--show-cmd', action='store_true',
    #                     help='Display cmd name of running process')
    # parser.add_argument(
    #     '-f', '--show-full-cmd', action='store_true',
    #     help='Display full command and cpu stats of running process'
    # )
    parser.add_argument('-u', '--show-user', action='store_true',
                        help='Display username of running process')
    parser.add_argument('-p', '--show-pid', action='store_true',
                        help='Display PID of running process')
    # parser.add_argument('-F', '--show-fan-speed', '--show-fan',
    #                     action='store_true', help='Display GPU fan speed')
    parser.add_argument('--json', action='store_true', default=False,
                        help='Print all the information in JSON format')
    parser.add_argument('-v', '--version', action='version',
                        version=('gpustat %s' % __version__))
    # parser.add_argument(
    #     '-P', '--show-power', nargs='?', const='draw,limit',
    #     choices=['', 'draw', 'limit', 'draw,limit', 'limit,draw'],
    #     help='Show GPU power usage or draw (and/or limit)'
    # )
    parser.add_argument(
        '-i', '--interval', '--watch', nargs='?', type=float, default=0,
        help='Use watch mode if given; seconds to wait between updates'
    )
    parser.add_argument(
        '--no-header', dest='show_header', action='store_false', default=True,
        help='Suppress header message'
    )
    parser.add_argument(
        '--gpuname-width', type=int, default=16,
        help='The minimum column width of GPU names, defaults to 16'
    )
    parser.add_argument(
        '--debug', action='store_true', default=False,
        help='Allow to print additional informations for debugging.'
    )
    args = parser.parse_args(argv[1:])
    # if args.show_all:
    #     #args.show_cmd = True
    args.show_user = True
    args.show_pid = True
        #args.show_fan_speed = True
        #args.show_power = 'draw,limit'
    #del args.show_all

    if args.interval is None:  # with default value
        args.interval = 1.0
    if args.interval > 0:
        args.interval = max(0.1, args.interval)
        if args.json:
            sys.stderr.write("Error: --json and --interval/-i "
                             "can't be used together.\n")
            sys.exit(1)

        loop_gpustat(**vars(args))
    else:
        del args.interval
        print_gpustat(**vars(args))
        sys.stdout = open('file', 'w')
        print_gpustat(**vars(args))


if __name__ == '__main__':
    main(*sys.argv)
