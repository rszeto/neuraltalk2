import matplotlib.pyplot as plt
import numpy as np
import json
from pprint import pprint

def loadJson(fileLoc):
    data_file = open(fileLoc)
    return json.load(data_file)
    
def pointsToPlot(points):
    points = sorted(points)
    xvals = []
    yvals = []
    for x,y in points:
        xvals.append(x)
        yvals.append(y)
    return {'x':xvals, 'y':yvals}

def valLossHistToPlot(val_loss_history):
    points = [(int(x),y) for x,y in val_loss_history.iteritems()]
    return pointsToPlot(points)

modelA = '../neuraltalk2_nojoints/checkpoints/model_idcoco_humanNoBB_baseline.json'

print 'Model A: ' + modelA

plt.figure(1)
dataA = loadJson(modelA)
plotDataA = valLossHistToPlot(dataA['val_loss_history'])
plt.plot(plotDataA['x'], plotDataA['y'], label='A')

# Find min iteration of both methods
minIter = plotDataA['x'][-1]
plt.xlim([0, minIter])
plt.xlabel('# iterations')
plt.ylabel('Validation loss')
# plt.title('Validation loss difference between A and B')
# plt.show()
plt.savefig('valid_loss.png')

plt.figure(2)
langDataA = [(int(x), y['CIDEr']) for x,y in dataA['val_lang_stats_history'].iteritems()]
langPlotDataA = pointsToPlot(langDataA)
plt.plot(langPlotDataA['x'], langPlotDataA['y'], label='A')

plt.xlim([0, minIter])
plt.xlabel('# iterations')
plt.ylabel('CIDEr score')
# plt.title('CIDEr difference between A and B')
plt.show()
plt.savefig('cider_score.png')
