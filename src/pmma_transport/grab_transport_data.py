import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from xlrd import open_workbook
import xlrd


def num(s):
    try:
        return int(s)
    except ValueError:
        return np.float64(s)


class DiffData:
    def __init__(self, path, file):
        #  initialize data... collect all the mess from the spreadsheet
        wb = open_workbook(path + file)
        s = wb.sheets()[0]
        self.sample_id = s.cell(0, 1).value
        self.material = s.cell(1, 1).value.lower()
        self.material = self.material.replace(" ", "")
        self.thickness = s.cell(1, 3).value.lower()
        self.thickness = self.thickness.replace(" ", "")
        self.thickness = self.thickness.replace("mil", "")
        self.thickness = num(self.thickness)
        self.temp = s.cell(9, 1).value
        self.RH = s.cell(9, 3).value / 100.0
        self.instrument = s.cell(8, 1).value.lower()
        self.instrument = self.instrument.replace(" ", "")
        self.comments = s.cell(5, 1).value
        self.count = 1
        self.description = s.cell(2, 1).value
        self.path = path
        self.fname = file
        self.entry = ''

        nr = s.nrows
        nc = s.ncols

        #  this stuff locates when the measurments end
        drs = 0
        dre = nr-1
        insidevals = False
        for row in range(12, nr):
            cell = s.cell(row, 3)
            if(cell.value == 'Raw'):
                drs = row+2
                insidevals = True
            if(insidevals):
                if(cell == xlrd.XL_CELL_EMPTY):
                    insidevals = False
                    dre = row
                    break

        times = s.col_slice(0)
        dat = s.col_slice(4)
        datsub = dat[drs:dre-1]
        timessub = times[drs:dre-1]

        times = np.zeros([(dre-drs)-1])
        transp = np.zeros([(dre-drs)-1])
        i = 0
        for t, d in zip(timessub, datsub):
            #  print t,d
            times[i] = t.value*3600
            transp[i] = d.value/(1e3*24*3600)
            i += 1

        self.ts = times
        self.js = transp
        self.sigma = np.std(self.js[self.ts > (self.ts[-1]-4000)])
