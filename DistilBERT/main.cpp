#include <torch/torch.h>
#include <torch/script.h> 
#include <vector>
int main()
{
	//use cuda
	auto device = torch::Device(torch::kCPU, -1);
	//load picture
	
	std::vector<torch::jit::IValue> input_id {101, 2002, 2056, 1996, 9440, 2121, 7903, 2063, 11345, 2449, 2987, 1005, 1056, 4906, 1996, 2194, 1005, 1055, 2146, 1011, 2744, 3930, 5656, 1012, 102, 1000, 1996, 9440, 2121, 7903, 2063, 11345, 2449, 2515, 2025, 4906, 2256, 2146, 1011, 2744, 3930, 5656, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	std::vector<torch::jit::IValue> input_mask  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	std::vector<torch::jit::IValue> segment_id {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	int label_id[] = {1};
	int label=1;


	float token[] = {101,  2040,  2001,  3958, 27227,  1029,   102,  3958,   103,  2001,  1037, 13997, 11510,   102};
	float segment2[] = {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1};
	auto options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU,-1);
	torch::Tensor token_ = torch::from_blob(token, {sizeof(token)}, options);
	torch::Tensor segment2_ = torch::from_blob(segment2, {sizeof(segment2)}, options);

	
	

	// float input_id[] =  {101, 2002, 2056, 1996, 9440, 2121, 7903, 2063, 11345, 2449, 2987, 1005, 1056, 4906, 1996, 2194, 1005, 1055, 2146, 1011, 2744, 3930, 5656, 1012, 102, 1000, 1996, 9440, 2121, 7903, 2063, 11345, 2449, 2515, 2025, 4906, 2256, 2146, 1011, 2744, 3930, 5656, 1012, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	// float input_mask[] =  {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	// float segment_id[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
	// torch::Tensor input_ids = torch::from_blob(input_id, {sizeof(input_id)}, options);
	// torch::Tensor input_masks = torch::from_blob(input_mask, {sizeof(input_mask)}, options);
	// torch::Tensor segment_ids = torch::from_blob(segment_id, {sizeof(segment_id)}, options);

	




	// torch::Tensor label_ids = torch::from_blob(label_id, {sizeof(label_id)}, options);
	
	// auto label_ids2 = label_ids.to(device).numpy_T();

	// auto 
	//resize size
	// cv::resize(image, image, cv::Size(224, 224));
	//translate to tensor
	// auto input_tensor = torch::from_blob(image.data, { image.rows, image.cols, 3 }, torch::kByte).permute({ 2, 0, 1 }).unsqueeze(0).to(torch::kFloat32) / 225.0;
	// //load model
	auto model = torch::jit::load("/home/AnonymousELF/My_Libtorch/DistilBERT/traced_bert_cpu.pt");
	model.to(device);
	model.eval();
	//forward pass
	
	// auto output = model.forward({token_, segment2_});
	auto output = model.forward({input_id , input_mask ,segment_id});
	// auto output = torch::argmax(logits, 1);
	// output = torch::softmax(output, 1);
	// std::cout<<torch::argmax(label_ids);
	// std::cout<< sizeof(output) <<" "<< sizeof(label_ids);
	// if( output == label_ids2 ) std::cout << "Inferenced Same!\n";
	// else std::cout << "Inferenced Different!\n";



	// output = torch::softmax(output, 1);
	// std::cout << "model result is " << torch::argmax(output) << "\nclass，precise value" << output.max() << std::endl;
	// std::cout<<"Successfully Loaded!\n";
	return 0;
}
