class Logger:
    def __init__(self, verbosity_level):
        # higher verbosity level means more verbose
        self.verbosity_level = verbosity_level

    def get_verbosity_level(self):
        return self.verbosity_level

    def log(self, verbosity: int, content: str):
        if verbosity <= self.verbosity_level:
            self.force_log(content)

    def force_log(self, content: str):
        raise NotImplementedError()


class StdoutLogger(Logger):
    def __init__(self, verbosity_level=0):
        super().__init__(verbosity_level)

    def force_log(self, content: str):
        print(content, flush=True)
