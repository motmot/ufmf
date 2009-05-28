import motmot.ufmf.traited_ufmf_writer as tuw
import tempfile, os

def test_empty():
    filename = tempfile.mkstemp()[1]
    w = 640
    h = 480
    try:
        ufmf=tuw.UfmfWriter(filename=filename,
                            max_width=w,
                            max_height=h)
        ufmf.close()
    finally:
        os.unlink(filename)
