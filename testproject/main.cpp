#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h> 

int main()
{
	//use cuda
	auto device = torch::Device(torch::kCPU, 0);
	//load picture
	auto image = cv::imread("/home/AnonymousELF/My_Libtorch/testproject/flower.jpg");
	//resize size
	cv::resize(image, image, cv::Size(224, 224));
	//translate to tensor
	auto input_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }).unsqueeze(0).to(torch::kFloat32) / 225.0;
	// //load model
	auto model = torch::jit::load("/home/AnonymousELF/My_Libtorch/testproject/resnet34.pt");
	model.to(device);
	model.eval();
	//forward pass
	auto output = model.forward({ input_tensor.to(device) }).toTensor();
	output = torch::softmax(output, 1);
	std::cout << "model result is " << torch::argmax(output) << "\nclass，precise value" << output.max() << std::endl;
	return 0;
}