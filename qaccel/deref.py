from IPython.parallel import Client

client = Client()


class Deref:
    """Dereference ipython parallel results.

    If parallel, use an IPython.parallel client to retrieve a result.
    Otherwise, the object is the result
    """

    def __init__(self, parallel):
        self.ll = parallel

    def __call__(self, var):
        if self.ll:
            return client.get_result(var).get()
        else:
            return var
