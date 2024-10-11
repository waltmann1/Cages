class Loggable(object):

    def __init__(self, log_list):
        self.log_list = log_list

    def add_to_logger(self):

        print("logger shit", self.log_values)
        if hasattr(self, 'log_values') and self.log_list is not None:
            if isinstance(self.log_values, list):
                if isinstance(self.log_values[0], str):
                    self.log_list += self.log_values
            else:
                raise TypeError('log_values must be a list')

    def remove_from_logger(self):

        for thing in self.log_values:
            try:
                self.log_list.remove(thing)
            except ValueError:
                pass


