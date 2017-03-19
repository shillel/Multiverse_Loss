require 'image'
require 'nn'
local function hflip(x)
   return (torch.random(0,1) == 1) and x or image.hflip(x)  --  (x) ? a : b
end

local function randomcrop(im , pad, randomcrop_type)
   if randomcrop_type == 'reflection' then
      -- Each feature map of a given input is padded with the replication of the input boundary
      module = nn.SpatialReflectionPadding(pad,pad,pad,pad):float() 
   elseif randomcrop_type == 'zero' then
      -- Each feature map of a given input is padded with specified number of zeros.
	  -- If padding values are negative, then input is cropped.
      module = nn.SpatialZeroPadding(pad,pad,pad,pad):float()
   end
	
   local padded = module:forward(im:float())
   local x = torch.random(1,pad*2 + 1)
   local y = torch.random(1,pad*2 + 1)
   --image.save('img2ZeroPadded.jpg', padded)

   return padded:narrow(3,x,im:size(3)):narrow(2,y,im:size(2))
end

do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local permutation = torch.randperm(input:size(1))
      for i=1,input:size(1) do
        if 0 == permutation[i] % 4  then
		image.save('imghflipOriginal.jpg',input[i]:float()) 
		image.save('imghflip.jpg',hflip(input[i]:float())) 
	end
	if 1 == permutation[i] % 2  then
		image.save('imgCropReflectionOriginal.jpg',input[i]:float()) 
		image.save('imgCropReflection.jpg',randomcrop(input[i]:float(),10,'reflection')) 
		--randomcrop(input[i]:float(),10,'reflection')
	end
	if 2 == permutation[i] % 4  then
		image.save('imgCropZeroOriginal.jpg',input[i]:float()) 
		image.save('imgCropZero.jpg',randomcrop(input[i]:float(),10,'zero')) 
		--randomcrop(input[i]:float(),10,'zero') 
	end
      end
    end
    self.output:set(input:cuda())
    return self.output
  end
end
 

--[[
Flips image src horizontally (left<->right). 
If dst is provided, it is used to store the output image. 
Otherwise, returns a new res Tensor
]]




-- Test hflip
--local im = image.load('homer_80_80.jpg')
--image.save('imghflip.jpg', image.hflip(im))

-- Test reflection
--local im = image.load('homer_80_80.jpg')
--local I = randomcrop(im, 10, 'reflection')
--print(I:size(),im:size())
--image.save('img2ReflectionCrop.jpg', I)

-- Test zero padding with crop
--local im = image.load('homer_80_80.jpg')
--local I = randomcrop(im, 10, 'zero')
--print(I:size(),im:size())
--image.save('img2ZeroCrop.jpg', I)

