/* Shared Use License: This file is owned by Derivative Inc. (Derivative)
* and can only be used, and/or modified for use, in conjunction with
* Derivative's TouchDesigner software, and only if you are a licensee who has
* accepted Derivative's TouchDesigner license or assignment agreement
* (which also govern the use of this file). You may share or redistribute
* a modified version of this file provided the following conditions are met:
*
* 1. The shared file or redistribution must retain the information set out
* above and this list of conditions.
* 2. Derivative's name (Derivative Inc.) or its trademarks may not be used
* to endorse or promote products derived from this file without specific
* prior written permission from Derivative.
*/

#include "PyTorchTOP.h"

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <cmath>
#include <random>
#include <chrono>

#include <c10/cuda/CUDACachingAllocator.h>

// These functions are basic C function, which the DLL loader can find
// much easier than finding a C++ Class.
// The DLLEXPORT prefix is needed so the compile exports these functions from the .dll
// you are creating
extern "C"
{

DLLEXPORT
void
FillTOPPluginInfo(TOP_PluginInfo *info)
{
	// This must always be set to this constant
	info->apiVersion = TOPCPlusPlusAPIVersion;

	// Change this to change the executeMode behavior of this plugin.
	info->executeMode = TOP_ExecuteMode::CPUMemWriteOnly;
	//info->executeMode = TOP_ExecuteMode::CUDA;

	// The opType is the unique name for this TOP. It must start with a 
	// capital A-Z character, and all the following characters must lower case
	// or numbers (a-z, 0-9)
	info->customOPInfo.opType->setString("Pytorchtop");

	// The opLabel is the text that will show up in the OP Create Dialog
	info->customOPInfo.opLabel->setString("PyTorch TOP");

	// Will be turned into a 3 letter icon on the nodes
	info->customOPInfo.opIcon->setString("PTT");

	// Information about the author of this OP
	info->customOPInfo.authorName->setString("David Braun");
	info->customOPInfo.authorEmail->setString("github.com/DBraun");

	info->customOPInfo.minInputs = 1;
	info->customOPInfo.maxInputs = 1;
}

DLLEXPORT
TOP_CPlusPlusBase*
CreateTOPInstance(const OP_NodeInfo* info, TOP_Context* context)
{
	// Return a new instance of your class every time this is called.
	// It will be called once per TOP that is using the .dll
	return new PyTorchTOP(info);
}

DLLEXPORT
void
DestroyTOPInstance(TOP_CPlusPlusBase* instance, TOP_Context *context)
{
	// Delete the instance here, this will be called when
	// Touch is shutting down, when the TOP using that instance is deleted, or
	// if the TOP loads a different DLL
	delete (PyTorchTOP*)instance;
}

};

// Undef this if you want to run an example that fills the data using threading
//#define THREADING_EXAMPLE


PyTorchTOP::PyTorchTOP(const OP_NodeInfo* info) : 
	myNodeInfo(info),
	myThread(nullptr),
	myThreadShouldExit(false)
{
	myExecuteCount = 0;
}

PyTorchTOP::~PyTorchTOP()
{
#ifdef THREADING_EXAMPLE
	if (myThread)
	{
		myThreadShouldExit.store(true);
		if (myThread->joinable())
		{
			myThread->join();
		}
		delete myThread;
	}
#endif
	// reset private vars that hold lots of memory.
	module = torch::jit::script::Module();
	torchinputs.clear();
	reuse_tensor = at::Tensor();
	// https://discuss.pytorch.org/t/how-to-manually-delete-free-a-tensor-in-aten/64153/2
	c10::cuda::CUDACachingAllocator::emptyCache();
}

void
PyTorchTOP::getGeneralInfo(TOP_GeneralInfo* ginfo, const OP_Inputs* inputs, void* reserved1)
{
	ginfo->cookEveryFrame = false;
    ginfo->memPixelType = OP_CPUMemPixelType::RGBA32Float;
}

bool
PyTorchTOP::getOutputFormat(TOP_OutputFormat* format, const OP_Inputs* inputs, void* reserved1)
{

	const OP_TOPInput* input = inputs->getInputTOP(0);
	if (!input) {
		return false;
	}

	inputs->getInputTOP(0);

	format->width = input->width;
	format->height = input->height;

	return true;
}

bool PyTorchTOP::checkModelFile(const char* newModelFilePath) {

	if (emptyString.compare(newModelFilePath) == 0) {

		myErrors = 1;
		return false;
	}

	else if (!strcmp(loadedModelFilePath.c_str(), newModelFilePath)) {
		return true;
	}

	try {
		// Deserialize the ScriptModule from a file using torch::jit::load().
		// module = torch::jit::load("udnie_1280x720.pt");
		module = torch::jit::load(newModelFilePath);

		// Save the name of the model we just successfully loaded.
		loadedModelFilePath = newModelFilePath;
	}
	catch (const c10::Error & e) {

		myErrors = 2;
		errorStream << e.msg();
		// std::cerr << "error loading the model\n";
		return false;
	}
	catch (...) {
		myErrors = 2;
		errorStream << "error loading the model\n";
		return false;
	}

	return true;
}

void
PyTorchTOP::execute(TOP_OutputFormatSpecs* output,
						const OP_Inputs* inputs,
						TOP_Context *context,
						void* reserved1)
{
	myExecuteCount++;
	myErrors = 0;
	errorStream.str("");

#ifdef THREADING_EXAMPLE
	mySettingsLock.lock();
#endif

#ifdef THREADING_EXAMPLE
	mySettingsLock.unlock();

	// This syncs up the buffers in the frame queue with what the
	// node is offering
	myFrameQueue.sync(output);

	if (!myThread)
	{
		myThread = new std::thread(
			[this]()
			{
				std::random_device rd;
				std::mt19937 mt(rd());

				// We are going to generate new frame data at irregular interval
				std::uniform_real_distribution<double> dist(10.0, 40.0);

				// Exit when our owner tells us to
				while (!this->myThreadShouldExit)
				{
					int width, height;
					void *buf = this->myFrameQueue.getBufferForUpdate(&width, &height);

					// If there is a buffer to update
					if (buf)
					{
						this->mySettingsLock.lock();
						double step = myStep;
						double brightness = myBrightness;
						this->mySettingsLock.unlock();

						CPUMemoryTOP::fillBuffer((float*)buf, width, height, step, brightness);


						this->myFrameQueue.updateComplete();
					}

					// Sleep for a random number of milliseconds
					std::this_thread::sleep_for(std::chrono::milliseconds(long long(dist(mt))));

				}
			});
	}

	// Tries to assign a buffer to be uploaded to the TOP
	myFrameQueue.sendBufferForUpload(output);

#else

	int textureMemoryLocation = 0;
	float* mem = (float*)output->cpuPixelData[textureMemoryLocation];

	const char* Imagedownload = inputs->getParString("Imagedownload");
	imageDownloadOptions.cpuMemPixelType = OP_CPUMemPixelType::BGRA8Fixed;
	if (!strcmp(Imagedownload, "Delayed")) {
		imageDownloadOptions.downloadType = OP_TOPInputDownloadType::Delayed;
	}
	else {
		imageDownloadOptions.downloadType = OP_TOPInputDownloadType::Instant;
	}

	const OP_TOPInput* input = inputs->getInputTOP(0);
	if (!input) {
		myErrors = 3;
		return;
	}

	void* videoSrc = (void*)inputs->getTOPDataInCPUMemory(input, &imageDownloadOptions);

	if (!videoSrc) {
		myErrors = 3;
		return;
	}

	const char* Modelfile = inputs->getParFilePath("Modelfile");

	if (!checkModelFile(Modelfile)) {
		return;
	}

	int WIDTH = output->width;
	int HEIGHT = output->height;

	if (!hasSetup) {
		reuse_tensor = torch::ones({ 1, HEIGHT, WIDTH, 4 }, device = device);
		hasSetup = true;
	} else if (reuse_tensor.sizes()[1] != HEIGHT || reuse_tensor.sizes()[2] != WIDTH) {
		reuse_tensor = torch::ones({ 1, HEIGHT, WIDTH, 4 }, device = device);
	}
	
	//std::cout << "cuda available? " << torch::cuda::is_available() << std::endl;

	// https://github.com/pytorch/pytorch/blob/master/torch/csrc/api/include/torch/types.h
	const at::TensorOptions tensorOptions = at::TensorOptions().dtype(torch::kByte);

	reuse_tensor.copy_(torch::from_blob(videoSrc, { 1, HEIGHT, WIDTH, 4 }, tensorOptions));

	//std::cout << reuse_tensor.sizes() << '\n';

	torchinputs.clear();
	// The thing we push back must be of format {1, 3, 720, 1280} because it's 1 image, 3 color channels, 720 rows, 1280 columns
	// Since we're starting with something that's {1, 720, 1280, 4},
	// we'll first narrow the last dimension to three channels rather than 4.
	// Then permute.
	torchinputs.push_back(reuse_tensor.narrow(3, 0, 3).permute({ 0,3,1,2 }));

	// Execute the model and turn its output into a tensor.
	// Putting it on the CPU at the end is what allows us to access values with a pointer.
	at::Tensor torchoutput = module.forward(torchinputs).toTensor().div(255.).to(torch::kCPU);
	torchinputs.clear();

	fillBuffer(mem, output->width, output->height, torchoutput);
	// Tell the TOP which buffer to upload. In this simple example we are always filling and uploading buffer 0
	output->newCPUPixelDataLocation = textureMemoryLocation;

#endif

}


void
PyTorchTOP::fillBuffer(float* mem, int width, int height, at::Tensor tensordata) {
	// https://discuss.pytorch.org/t/iterating-over-tensor-in-c/60333/2
	// https://discuss.pytorch.org/t/libtorch-c-convert-a-tensor-to-cv-mat-single-channel/47701/4
	//std::cout << "size " << tensordata.sizes() << '\n';

	float* ptr = (float*)tensordata.data_ptr();
	//std::cout << "is_contiguous " << tensordata.is_contiguous() << '\n';

	// iterate by each dimension
	for (int colorChan = 0; colorChan < tensordata.sizes()[1]; ++colorChan)
	{
		for (int y = 0; y < tensordata.sizes()[2]; ++y)
		{
			for (int x = 0; x < tensordata.sizes()[3]; ++x)
			{
				//float* pixel = &mem[4 * (y * width + x)];
				//pixel[colorChan] = (*ptr++)/255.;
				mem[4 * (y * width + x) + colorChan] = *ptr++;
			}
		}
	}

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			// float* pixel = &mem[4 * (y * width + x)];
			// pixel[0] = 0.;
			// pixel[1] = 0.;
			// pixel[2] = 0.;
			// pixel[3] = 1.; // make the alpha channel 1.
			mem[4 * (y * width + x) + 3] = 1.; // optimized for just making alpha=1.
		}
	}

}

// You can use this function to put the node into a error state
// by calling setSting() on 'error' with a non empty string.
// Leave 'error' unchanged to not go into error state.
void PyTorchTOP::getErrorString(OP_String* error, void* reserved1) {

	switch (myErrors) {
	case 0:
		// no errors. must set string to blank.
		error->setString("");
		break;
	case 1:
		error->setString("The requested model file path is blank.");
		break;
	case 2:
		error->setString(errorStream.str().c_str());
		break;
	case 3:
		error->setString("You must connect an input TOP.");
		break;
	}

	}

// You can use this function to put the node into a warning state
// by calling setSting() on 'warning' with a non empty string.
// Leave 'warning' unchanged to not go into warning state.
void PyTorchTOP::getWarningString(OP_String* warning, void* reserved1)
{
}

// Use this function to return some text that will show up in the
// info popup (when you middle click on a node)
// call setString() on info and give it some info if desired.
void PyTorchTOP::getInfoPopupString(OP_String* info, void* reserved1)
{
}

int32_t
PyTorchTOP::getNumInfoCHOPChans(void* reserved1)
{
	// We return the number of channel we want to output to any Info CHOP
	// connected to the TOP. In this example we are just going to send one channel.
	return 1;
}

void
PyTorchTOP::getInfoCHOPChan(int32_t index, OP_InfoCHOPChan* chan, void* reserved1)
{
	// This function will be called once for each channel we said we'd want to return
	// In this example it'll only be called once.

	if (index == 0)
	{
		chan->name->setString("executeCount");
		chan->value = (float)myExecuteCount;
	}

}

bool
PyTorchTOP::getInfoDATSize(OP_InfoDATSize* infoSize, void* reserved1)
{
	infoSize->rows = 1;
	infoSize->cols = 2;
	// Setting this to false means we'll be assigning values to the table
	// one row at a time. True means we'll do it one column at a time.
	infoSize->byColumn = false;
	return true;
}

void
PyTorchTOP::getInfoDATEntries(int32_t index,
	int32_t nEntries,
	OP_InfoDATEntries* entries,
	void* reserved1)
{
	char tempBuffer[4096];

	if (index == 0)
	{
		// Set the value for the first column
#ifdef _WIN32
		strcpy_s(tempBuffer, "executeCount");
#else // macOS
		strlcpy(tempBuffer, "executeCount", sizeof(tempBuffer));
#endif
		entries->values[0]->setString(tempBuffer);

		// Set the value for the second column
#ifdef _WIN32
		sprintf_s(tempBuffer, "%d", myExecuteCount);
#else // macOS
		snprintf(tempBuffer, sizeof(tempBuffer), "%d", myExecuteCount);
#endif
		entries->values[1]->setString(tempBuffer);
	}
}

void
PyTorchTOP::setupParameters(OP_ParameterManager* manager, void* reserved1)
{
	//// brightness
	//{
	//	OP_NumericParameter	np;

	//	np.name = "Brightness";
	//	np.label = "Brightness";
	//	np.defaultValues[0] = 1.0;

	//	np.minSliders[0] =  0.0;
	//	np.maxSliders[0] =  1.0;

	//	np.minValues[0] = 0.0;
	//	np.maxValues[0] = 1.0;

	//	np.clampMins[0] = true;
	//	np.clampMaxes[0] = true;
	//	
	//	OP_ParAppendResult res = manager->appendFloat(np);
	//	assert(res == OP_ParAppendResult::Success);
	//}

	//// speed
	//{
	//	OP_NumericParameter	np;

	//	np.name = "Speed";
	//	np.label = "Speed";
	//	np.defaultValues[0] = 1.0;
	//	np.minSliders[0] = -10.0;
	//	np.maxSliders[0] =  10.0;
	//	
	//	OP_ParAppendResult res = manager->appendFloat(np);
	//	assert(res == OP_ParAppendResult::Success);
	//}

	//// pulse
	//{
	//	OP_NumericParameter	np;

	//	np.name = "Reset";
	//	np.label = "Reset";
	//	
	//	OP_ParAppendResult res = manager->appendPulse(np);
	//	assert(res == OP_ParAppendResult::Success);
	//}

	// Image download type
	{
		OP_StringParameter	sp;

		sp.name = "Imagedownload";
		sp.label = "Image Download";

		sp.defaultValue = "Instant";

		const char* names[] = { "Instant", "Delayed" };
		const char* labels[] = { "Instant", "Delayed" };

		OP_ParAppendResult res = manager->appendMenu(sp, 2, names, labels);
		assert(res == OP_ParAppendResult::Success);
	}

	// Model File
	{
		OP_StringParameter sp;

		sp.name = "Modelfile";
		sp.label = "Model File";
		sp.defaultValue = "";
		OP_ParAppendResult res = manager->appendFile(sp);
		assert(res == OP_ParAppendResult::Success);
	}

}

void
PyTorchTOP::pulsePressed(const char* name, void* reserved1)
{
	//if (!strcmp(name, "Reset"))
	//{

	//}
}
