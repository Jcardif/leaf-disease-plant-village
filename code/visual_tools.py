# modified from https://github.com/utkuozbulak/pytorch-cnn-visualizations
# to adjust to resnet101
import itertools
import cv2
from shutil import copyfile
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import os
import math
import numpy as np
from ImageFolderData import default_loader
irange = range
def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=255, predict_list=None, target_list=None, prob_list=None, write_prob=True):
    """Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.
    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.view(1, tensor.size(0), tensor.size(1), tensor.size(2))

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, t.min(), t.max())

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze()

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    #grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    grid = tensor.new(height * ymaps + padding, width * xmaps + padding, 3).fill_(pad_value)
    writen_words = []
    writen_probs = []
    if write_prob==True:
        for prob in prob_list:
            writen_probs.append(str(prob))
    if predict_list is not None:
        if target_list is not None:
            for n, predict in enumerate(predict_list):
                writen_words.append(str(predict)+': '+str(target_list[n]))
        else:
            for n, predict in enumerate(predict_list):
                writen_words.append(str(predict))

    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            tensor_np = tensor[k].cpu().numpy().transpose((1, 2, 0)).copy()
            if x % 5 == 0:# origin pics, gradcam, gradcamplus
                tensor_with_label = cv2.putText(tensor_np,writen_words[(y*xmaps+x)//5], (0,40), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(255,255,255),3)
                if write_prob==True:
                    tensor_with_label = cv2.putText(tensor_np,writen_probs[(y*xmaps+x)//5], (0,215), cv2.FONT_HERSHEY_SIMPLEX, 1.5,(255,255,255),3)
                tensor_with_label = torch.from_numpy(tensor_with_label)
            else:
                tensor_with_label = torch.from_numpy(tensor_np) #.transpose((2, 0, 1))
            
            grid.narrow(0, y * height + padding, height - padding)\
                .narrow(1, x * width + padding, width - padding)\
                .copy_(tensor_with_label)
            k = k + 1
    return grid

class CamExtractor():
    """
        Extracts cam features from the model
    """
    def __init__(self, retrained_model, target_layer, base_model=None):
        self.target_layer = target_layer
        self.gradients = None
        retrained_model.eval()        
        self.retrained_model = retrained_model
        if base_model is not None:
            base_model.eval()
        self.base_model = base_model
    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
            Does a forward pass on convolutions, hooks the function at the last convolutional layer
            output of resnet model
        """
        conv_output = None
        self.retrained_model.eval()
        if self.base_model is not None:
            self.base_model.eval()
            
            iteritems = itertools.chain(self.base_model.module._modules.items()
                                           , self.retrained_model.module._modules.items())
            #test_out = self.retrained_model(self.base_model(x))
        else:
            iteritems = self.retrained_model.module._modules.items()
            #test_out = self.retrained_model(x)
        assert self.target_layer.find('layer') != -1
        layer, bottlenecknum, modulename = self.target_layer.split('/')
        for module_pos, module in iteritems:
            if module_pos != 'fc':
                x = module(x)
                #print(module_pos)
                #"""
                if module_pos == layer:
                    #x.register_hook(self.save_gradient)
                    x.retain_grad()
                    conv_output = x

                                          
        return conv_output, x#, test_out

    def forward_pass(self, x):
        """
            Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        #print(conv_output)
        x = x.view(x.size(0), -1)  # Flatten

        # Forward pass on the classifier
        x = self.retrained_model.module.fc(x)

        #print(x.data)
        #print(test_out.data)
        #assert x.data.cpu().numpy().all() == test_out.data.cpu().numpy().all()
        
        return conv_output, x

class GradCamPlus():
    #https://github.com/adityac94/Grad_CAM_plus_plus/blob/master/misc/utils.py
    def __init__(self, retrained_model, target_layer, base_model=None, usecuda=True):
        self.retrained_model = retrained_model
        self.base_model = base_model
        if self.base_model is not None:
            self.base_model.eval()
        self.retrained_model.eval()
        self.usecuda = usecuda
        # Define extractor
        self.extractor = CamExtractor(self.retrained_model, target_layer, self.base_model)
    def generate_cam(self, input_image, target_index=None, draw_acc_pred=True):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        pred_index = np.argmax(model_output.data.cpu().numpy())
        if target_index is None or (pred_index != target_index and draw_acc_pred):
            target_index = pred_index
            
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        if self.usecuda:
            one_hot_output = one_hot_output.cuda()
        one_hot_output[0][target_index] = float(model_output[0][target_index].data.cpu().numpy())#1
        # Zero grads
        if self.base_model is not None:
            self.base_model.zero_grad()
        self.retrained_model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True, create_graph=True)
        # Get hooked gradients
        guided_gradientsv = conv_output.grad#self.extractor.gradients #torch.Size([1, 2048, 7, 7])
        guided_gradients = guided_gradientsv.data.cpu().numpy()[0]
        # Get convolution outputs
        target = conv_output.data.cpu().numpy()[0]
        #print('gradcamplus')
        #print(one_hot_output)
        #print(np.exp(one_hot_output))
        #first_derivative
        first_derivative = np.exp(one_hot_output)[0][target_index]*guided_gradients
        """
        #second_derivative
        second_derivative = np.exp(one_hot_output)[0][target_index]*guided_gradients*guided_gradients#+np.exp(one_hot_output)[0][target_index]*guided_2gradients 

        #triple_derivative
        triple_derivative = np.exp(one_hot_output)[0][target_index]*guided_gradients*guided_gradients*guided_gradients#+np.exp(one_hot_output)[0][target_index]*guided_2gradients*guided_gradients+np.exp(one_hot_output)[0][target_index]*guided_gradients*guided_2gradients+np.exp(one_hot_output)[0][target_index]*guided_gradients*guided_2gradients+np.exp(one_hot_output)[0][target_index]*guided_3gradients
        #print(first_derivative.shape) # (2048, 7, 7)
        #print(target.shape) # (2048, 7, 7)
        global_sum = np.sum(target, axis=(1,2))
        #print(global_sum.shape) # (2048,)
        alpha_num = second_derivative
        alpha_denom = second_derivative*2.0 + triple_derivative*global_sum.reshape((first_derivative.shape[0],1,1))
        #print(alpha_denom.shape) # (2048, 7, 7)
        """
        global_sum = np.sum(target, axis=(1,2))
        alpha_num = 1
        alpha_denom = 2.0 + first_derivative*global_sum.reshape((first_derivative.shape[0],1,1))
        alpha_denom = np.where(alpha_denom != 0.0, alpha_denom, np.ones(alpha_denom.shape))
        alphas = alpha_num/alpha_denom

        weights = np.maximum(first_derivative, 0.0)
        #print(weights.shape)#(2048, 7, 7)
        #normalizing the alphas
        #print(alphas.shape) #(2048, 7, 7)
        alpha_normalization_constant = np.sum(alphas, axis=(1,2))
        #print(alpha_normalization_constant.shape) ## (2048,)
        alphas /= alpha_normalization_constant.reshape((first_derivative.shape[0], 1, 1))
        #print('camplus alphas')
        #print(alphas)
        deep_linearization_weights = np.sum((weights*alphas).reshape((first_derivative.shape[0], -1)),axis=1)
        #print(deep_linearization_weights.shape)# (2048,)
        # Create empty numpy array for cam
        grad_CAM_map = np.zeros(target.shape[1:], dtype=np.float32)#np.ones(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        #print('camplus weights')
        #print(deep_linearization_weights)
        for i, w in enumerate(deep_linearization_weights):
            grad_CAM_map += w * target[i, :, :]
        #print(grad_CAM_map.shape) #(7, 7)
        # Passing through ReLU
        cam = np.maximum(grad_CAM_map, 0)#relu
        cam = cam / cam.max() # scale 0 to 1.0  #(cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = cv2.resize(cam, (224, 224))
        
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        #print(cam)
        return cam, model_output
    
class GradCam():
    """
        Produces class activation map
    """
    def __init__(self, retrained_model, target_layer, base_model, usecuda=True):
        self.base_model = base_model
        self.retrained_model = retrained_model
        if self.base_model:
            self.base_model.eval()
        self.retrained_model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.retrained_model, target_layer, self.base_model)
        self.usecuda = usecuda

    def generate_cam(self, input_image, target_index=None, draw_feature_num=7, draw_acc_pred=True, file_name_to_export=''):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        # model_output is the final output of the model (1, 1000)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        pred_index = np.argmax(model_output.data.cpu().numpy())
        if target_index is None or (pred_index != target_index and draw_acc_pred):
            target_index = pred_index
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        if self.usecuda:
            one_hot_output = one_hot_output.cuda()
        one_hot_output[0][target_index] = float(model_output[0][target_index].data.cpu().numpy()) #1
        # Zero grads
        if self.base_model:
            self.base_model.zero_grad()
        self.retrained_model.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = conv_output.grad.data.cpu().numpy()[0]#self.extractor.gradients.data.cpu().numpy()[0]
        # Get convolution outputs
        target = conv_output.data.cpu().numpy()[0]
        # Get weights from gradients
        weights = np.mean(guided_gradients, axis=(1, 2))  # Take averages for each gradient
        #print('grad-cam weight: ')
        #print(weights)
        # Create empty numpy array for cam
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        
        cam = np.maximum(cam, 0)#relu
        cam = cam / cam.max() # scale 0 to 1.0  #(cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        cam = cv2.resize(cam, (224, 224))
        cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        args_sorted_weights = None
        if draw_feature_num != 0:
            args_sorted_weights = np.argsort(weights)[::-1]
            for n, i in enumerate(args_sorted_weights):
                if n == draw_feature_num:
                    break
                feature_map= cv2.resize(target[i,:,:], (224, 224))
                feature_map /= np.max(feature_map)
                feature_map = np.uint8(feature_map * 255)
                save_features_on_image(feature_map, file_name_to_export+'_'+str(i)+'_'+str(n)+'_last_conv_out')
        
        return cam, pred_index, args_sorted_weights
    
class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, retrained_model, processed_im, target_class=None, draw_acc_pred=True, base_model=None, usecuda=True):
        self.base_model = base_model
        self.retrained_model = retrained_model
        self.input_image = processed_im
        self.target_class = target_class
        self.gradients = None
        self.draw_acc_pred = draw_acc_pred
        # Put model in evaluation mode
        if self.base_model is not None:
            self.base_model.eval()
        self.retrained_model.eval()
        self.update_relus()
        self.hook_layers()
        self.usecuda = usecuda

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        if self.base_model is not None:
            first_layer = list(self.base_model.module._modules.items())[0][1]
        else:
            first_layer = list(self.retrained_model.module._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that it only returns positive gradients
        """
        def relu_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, changes it to zero
            """
            if isinstance(module, nn.ReLU):
                return (torch.clamp(grad_in[0], min=0.0),)
        # Loop through layers, hook up ReLUs with relu_hook_function
        if self.base_model is not None:
            iteritems = itertools.chain(self.base_model.module._modules.items()
                                           , self.retrained_model.module._modules.items())
        else:
            iteritems = self.retrained_model.module._modules.items()
        for pos, module in iteritems:
            if pos.find('layer') != -1:
                for _, mod in module._modules.items():
                    for po, mo in mod._modules.items():
                        if isinstance(mo, nn.ReLU):
                            mo.register_backward_hook(relu_hook_function)
            else:
                if isinstance(module, nn.ReLU):
                    module.register_backward_hook(relu_hook_function)

    def generate_gradients(self):
        # Forward pass
        if self.base_model is not None:
            model_output = self.retrained_model(self.base_model(self.input_image))
            self.base_model.zero_grad()
        else:
            model_output = self.retrained_model(self.input_image)
        # Zero gradients
        self.retrained_model.zero_grad()
        # Target for backprop
        pred_index = np.argmax(model_output.data.cpu().numpy())
        if self.target_class is None or (pred_index != self.target_class and self.draw_acc_pred):
            self.target_class = pred_index
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][self.target_class] = 1
        if self.usecuda:
            one_hot_output = one_hot_output.cuda()
        # Backward pass
        model_output.backward(gradient=one_hot_output)
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        gradients_as_arr = self.gradients.data.cpu().numpy()[0]
        return gradients_as_arr

def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    """
        Guided grad cam is just pointwise multiplication of cam mask and
        guided backprop mask

    Args:
        grad_cam_mask (np_arr): Class activation map mask
        guided_backprop_mask (np_arr):Guided backprop mask
    """
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    return cam_gb

def save_gradient_images(gradient, file_name):
    """
        Exports the original gradient image

    Args:
        gradient (np arr): Numpy array of the gradient with shape (3, 224, 224)
        file_name (str): File name to be exported
    """
    if not os.path.exists('results'):
        os.makedirs('results')
    gradient = gradient - gradient.min()
    gradient /= gradient.max()
    gradient = np.uint8(gradient * 255).transpose(1, 2, 0)
    path_to_file = os.path.join('results', file_name + '.jpg')
    # Convert RBG to GBR
    gradient = gradient[..., ::-1]
    cv2.imwrite(path_to_file, gradient)
    return gradient


def save_class_activation_on_image(org_img, activation_map, file_name):
    """
        Saves cam activation map and activation map on the original image

    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): activation map (grayscale) 0-255
        file_name (str): File name of the exported image
    """
    if not os.path.exists('results'):
        os.makedirs('results')
    # Grayscale activation map
    path_to_file = os.path.join('results', file_name+'_Cam_Grayscale.jpg')
    cv2.imwrite(path_to_file, activation_map)
    # Heatmap of activation map
    #print(activation_map)
    activation_heatmap = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)
    path_to_file = os.path.join('results', file_name+'_Cam_Heatmap.jpg')
    cv2.imwrite(path_to_file, activation_heatmap)
    # Heatmap on picture
    org_img = cv2.resize(org_img, (224, 224))
    img_with_heatmap = np.float32(activation_heatmap) + np.float32(org_img)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)
    path_to_file = os.path.join('results', file_name+'_Cam_On_Image.jpg')
    img_with_heatmap = np.uint8(255 * img_with_heatmap)
    cv2.imwrite(path_to_file, img_with_heatmap)
    return img_with_heatmap
    
def save_features_on_image(features_map,file_name):
    """
        Saves cam activation map and activation map on the original image

    Args:
        features_map: one of last convolutional layers' feature maps
        sorted_arguments (numpy arr): the index of the sorted weights of cam
        file_name (str): File name of the exported image
        num: how many feature maps we will save
    """
    if not os.path.exists('results'):
        os.makedirs('results')
    
    path_to_file = os.path.join('results', file_name+'_features_Grayscale.jpg')
    cv2.imwrite(path_to_file, features_map)
    # Heatmap of activation map
    activation_heatmap = cv2.applyColorMap(features_map, cv2.COLORMAP_JET)                                             
    path_to_file = os.path.join('results', file_name+'_features_Heatmap.jpg')
    cv2.imwrite(path_to_file, activation_heatmap)

def convert_to_grayscale(cv2im):
    """
        Converts 3d image to grayscale

    Args:
        cv2im (numpy arr): RGB image with shape (D,W,H)

    returns:
        grayscale_im (numpy_arr): Grayscale image with shape (1,W,D)
    """
    grayscale_im = np.sum(np.abs(cv2im), axis=0)
    im_max = np.percentile(grayscale_im, 99)
    im_min = np.min(grayscale_im)
    grayscale_im = (np.clip((grayscale_im - im_min) / (im_max - im_min), 0, 1))
    grayscale_im = np.expand_dims(grayscale_im, axis=0)
    return grayscale_im

def get_positive_negative_saliency(gradient):
    """
        Generates positive and negative saliency maps based on the gradient
    Args:
        gradient (numpy arr): Gradient of the operation to visualize
    returns:
        pos_saliency ( )
    """
    pos_saliency = (np.maximum(0, gradient) / gradient.max())
    neg_saliency = (np.maximum(0, -gradient) / -gradient.min())
    return pos_saliency, neg_saliency

def guided_gradcam(retrained_model, target_layer, img_dataset_loader=None, batch_ind=None, img_ind=None,  base_model=None, draw_feature_num=7, draw_acc_pred=True, data_sep='', given_img_path=None, classes_to_ind_dict=None, usecuda=True, has_target_labels=True):
    if given_img_path is not None:
        img_path = given_img_path
        if has_target_labels:
            cla_name = img_path.split('/')[-2]
            target_class = classes_to_ind_dict[cla_name]
        else:
            target_class = None
        print(img_path)
        #img = cv2.imread(img_path, 1)
        #img = cv2.resize(img, (224, 224))
        img = default_loader(img_path)

        # preprocessing is very important
        img = torchvision.transforms.functional.resize(img, (224, 224))
        prep_img = torchvision.transforms.functional.to_tensor(img)
        prep_img = torchvision.transforms.functional.normalize(torchvision.transforms.functional.to_tensor(img), [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        prep_img = torch.unsqueeze(prep_img, 0)
        # preprocessing
        
    else:
        for batch_i, (prep_imgs, img_paths, target_classes) in enumerate(img_dataset_loader):
            if batch_i == batch_ind:
                prep_img, img_path, target_class = prep_imgs[img_ind:img_ind+1], img_paths[img_ind], target_classes[img_ind]
                break
    original_image = cv2.imread(img_path, 1)
    
    #prep_img, img_path, target_class = next(iter(img_dataset_loader))
    #prep_img, img_path, target_class = prep_img[ind:ind+1], img_path[ind], target_class[ind]
    
    if usecuda:
        prep_img = prep_img.cuda(async=True)
    prep_imgv = Variable(prep_img, requires_grad=True)

    file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]+'_'+data_sep

    # Grad cam
    gcv2 = GradCam(retrained_model, target_layer, base_model, usecuda)
    # Generate cam mask
    cam, pred_index, args_sorted_weights = gcv2.generate_cam(prep_imgv, target_class, draw_feature_num, draw_acc_pred, file_name_to_export)
    print('cam')
    print(cam)

    if pred_index == target_class:
        file_name_to_export += '_classified_right'
    else:
        file_name_to_export += '_classified_wrong'
    
    
    # Gradcam++
    gcplusv2 = GradCamPlus(retrained_model, target_layer, base_model, usecuda)
    camplus, model_output = gcplusv2.generate_cam(prep_imgv, target_class, draw_acc_pred)
    print('camplus')
    print(camplus)
    img_with_gradcam_heatmap =save_class_activation_on_image(original_image, cam, file_name_to_export+'_'+data_sep)
    img_with_gradcamplus_heatmap = save_class_activation_on_image(original_image, camplus, file_name_to_export+'_'+data_sep+'_plus')
    print('Grad cam completed')
    
    # Guided backprop
    GBP = GuidedBackprop(retrained_model, prep_imgv, target_class, draw_acc_pred, base_model, usecuda)
    # Get gradients
    guided_grads = GBP.generate_gradients()
    # Save colored gradients
    save_gradient_images(guided_grads, file_name_to_export +'_'+data_sep+ '_Guided_BP_color')
    # Convert to grayscale
    grayscale_guided_grads = convert_to_grayscale(guided_grads)
    # Save grayscale gradients
    save_gradient_images(grayscale_guided_grads, file_name_to_export+'_'+data_sep + '_Guided_BP_gray')
    # Positive and negative saliency maps
    pos_sal, neg_sal = get_positive_negative_saliency(guided_grads)
    save_gradient_images(pos_sal, file_name_to_export+'_'+data_sep +  '_pos_sal')
    save_gradient_images(neg_sal, file_name_to_export+'_'+data_sep +  '_neg_sal')
    print('Guided backpropagation completed')
    
    
    # Guided Grad cam
    cam_gb = guided_grad_cam(cam, pos_sal)#, guided_grads)
    copyfile(img_path,os.path.join('results',file_name_to_export+'_'+data_sep+'_label_'+str(target_class)+'_pred_'+str(pred_index)+'.jpg'))
    save_gradient_images(cam_gb, file_name_to_export +'_'+data_sep+ '_GGrad_Cam')
    grayscale_cam_gb = convert_to_grayscale(cam_gb)
    grayscale_cam_gb_img = save_gradient_images(grayscale_cam_gb, file_name_to_export +'_'+data_sep+ '_GGrad_Cam_gray')
    #grayscale_cam_gb_minus_healthy = convert_to_grayscale(cam_gb_minus_healthy)
    #save_gradient_images(grayscale_cam_gb_minus_healthy, file_name_to_export + '_GGrad_Cam_gray_minus_healthy')
    print('Guided grad cam completed')
    
    # Guided Grad cam++
    cam_gb_plus = guided_grad_cam(camplus, pos_sal)#, guided_grads)
    ##copyfile(img_path,os.path.join('results',file_name_to_export+'_'+data_sep+'.jpg'))
    gradient_cam = save_gradient_images(cam_gb_plus, file_name_to_export +'_'+data_sep+ '_GGrad_Cam_Plus')
    grayscale_cam_gb_plus = convert_to_grayscale(cam_gb_plus)
    grayscale_cam_gb_plus_img = save_gradient_images(grayscale_cam_gb_plus, file_name_to_export +'_'+data_sep+ '_GGrad_Cam_Plus_gray')
    print('Guided grad cam++ completed') # #prep_img[0].cpu(),
    return torch.from_numpy(img_with_gradcam_heatmap.transpose(2, 0, 1)) , torch.from_numpy(img_with_gradcamplus_heatmap.transpose(2, 0, 1)), torch.from_numpy(cv2.resize(original_image,(224,224)).transpose(2, 0, 1)), torch.from_numpy(grayscale_cam_gb_img.transpose(2, 0, 1)[0]), torch.from_numpy(grayscale_cam_gb_plus_img.transpose(2, 0, 1)[0]), pred_index, target_class, model_output, args_sorted_weights
