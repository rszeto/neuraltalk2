local cjson = require 'cjson'
local utils = {}

-- Assume required if default_value is nil
function utils.getopt(opt, key, default_value)
  if default_value == nil and (opt == nil or opt[key] == nil) then
    error('error: required key ' .. key .. ' was not provided in an opt.')
  end
  if opt == nil then return default_value end
  local v = opt[key]
  if v == nil then v = default_value end
  return v
end

function utils.read_json(path)
  local file = io.open(path, 'r')
  local text = file:read()
  file:close()
  local info = cjson.decode(text)
  return info
end

function utils.write_json(path, j)
  -- API reference http://www.kyne.com.au/~mark/software/lua-cjson-manual.html#encode
  cjson.encode_sparse_array(true, 2, 10)
  local text = cjson.encode(j)
  local file = io.open(path, 'w')
  file:write(text)
  file:close()
end

-- dicts is a list of tables of k:v pairs, create a single
-- k:v table that has the mean of the v's for each k
-- assumes that all dicts have same keys always
function utils.dict_average(dicts)
  local dict = {}
  local n = 0
  for i,d in pairs(dicts) do
    for k,v in pairs(d) do
      if dict[k] == nil then dict[k] = 0 end
      dict[k] = dict[k] + v
    end
    n=n+1
  end
  for k,v in pairs(dict) do
    dict[k] = dict[k] / n -- produce the average
  end
  return dict
end

-- seriously this is kind of ridiculous
function utils.count_keys(t)
  local n = 0
  for k,v in pairs(t) do
    n = n + 1
  end
  return n
end

-- return average of all values in a table...
function utils.average_values(t)
  local n = 0
  local vsum = 0
  for k,v in pairs(t) do
    vsum = vsum + v
    n = n + 1
  end
  return vsum / n
end

-- Finds the maximum in a 2D tensor and its index because
-- it has to be unnecessarily complicated in Torch
function utils.max2(data)
  local maxVal1, maxInd1 = torch.max(data, 1)
  local maxVal2, maxInd2 = torch.max(maxVal1, 2)
  maxInd2 = maxInd2[1][1]
  maxInd1 = maxInd1[1][maxInd2]
  return maxVal2[1][1], torch.LongTensor({maxInd1, maxInd2})
end

-- Find the distances between each of the given points
-- points is a 2xN FloatTensor
function utils.allPairsDists(points)
  assert(torch.type(points) == 'torch.FloatTensor', 'points was not a FloatTensor')
  local numPoints = points:size(2)
  local numDists = numPoints*(numPoints-1)/2
  local pointDists = torch.FloatTensor(numDists)
  local i = 1
  for j=1,numPoints do
    for k=j+1,numPoints do
      pointDists[i] = torch.norm(points[{{}, j}] - points[{{}, k}])
      i = i+1
    end
  end
  return pointDists
end

return utils
