// src/compiled_model.ts
var CompiledModel = class {
  constructor(liteRtInterpreter, makeSignatureRunner, onDelete) {
    this.liteRtInterpreter = liteRtInterpreter;
    this.onDelete = onDelete;
    const tfliteInputs = this.liteRtInterpreter.inputs();
    for (let i = 0; i < tfliteInputs.size(); ++i) {
      const tensor = tfliteInputs.get(i);
      this.inputTensors.set(tensor.name(), tensor);
    }
    const signaturesVector = this.liteRtInterpreter.listSignatures();
    for (let i = 0; i < signaturesVector.size(); ++i) {
      const signatureName = signaturesVector.get(i);
      this.signatures[signatureName] = makeSignatureRunner(
        this.liteRtInterpreter.getSignatureRunner(signatureName)
      );
    }
    this.primarySignature = makeSignatureRunner(liteRtInterpreter);
    this.accelerator = this.primarySignature.accelerator;
  }
  inputTensors = /* @__PURE__ */ new Map();
  outputTensors = /* @__PURE__ */ new Map();
  signatures = {};
  primarySignature;
  accelerator;
  deleted = false;
  checkDeleted() {
    if (this.deleted) {
      throw new Error("Model has been deleted. Please reload the model.");
    }
  }
  run(inputOrSignatureName, maybeInput) {
    this.checkDeleted();
    if (typeof inputOrSignatureName === "string") {
      const signatureName = inputOrSignatureName;
      const input = maybeInput;
      const signature = this.signatures[signatureName];
      if (!signature) {
        const signatures = Object.keys(this.signatures).join(", ");
        throw new Error(`Signature '${signatureName}' not found in the model. Available signatures: ${signatures}`);
      }
      if (!input) {
        throw new Error(`No input provided for signature '${signatureName}'.`);
      }
      return signature.run(input);
    } else {
      return this.primarySignature.run(inputOrSignatureName);
    }
  }
  /**
   * Returns the input details for the primary signature.
   */
  getInputDetails() {
    this.checkDeleted();
    return this.primarySignature.getInputDetails();
  }
  /**
   * Returns the output details for the primary signature.
   */
  getOutputDetails() {
    this.checkDeleted();
    return this.primarySignature.getOutputDetails();
  }
  delete() {
    if (this.deleted) {
      return;
    }
    this.deleted = true;
    for (const signature of Object.values(this.signatures)) {
      signature.delete();
    }
    this.primarySignature.delete();
    for (const input of this.inputTensors.values()) {
      input.delete();
    }
    for (const output of this.outputTensors.values()) {
      output.delete();
    }
    this.liteRtInterpreter.delete();
    this.onDelete();
  }
};

// src/constants.ts
var ACCELERATORS = ["webgpu", "wasm"];
var DTYPE_TO_ARRAY_TYPE = Object.freeze({
  // noType is not supported.
  "float32": Float32Array,
  "int32": Int32Array
  // The following types are disabled until we support them in C++.
  /*
  'uint8': Uint8Array,
  // TODO(msoulanille): int64 is not supported yet because BigInt64Array makes
  // TFJS integration more complicated.
  // 'int64': BigInt64Array,
  // String is not supported.
  // TODO(msoulanille): bool will require special handling in C++.
  // TFJS WebGPU stores bool in a 32 bit integer.
  // However, tf.data() returns a Uint8Array.
  // Unclear if we should follow TFJS or whatever LiteRt xnnpack does.
  'bool': Uint8Array,
  'int16': Int16Array,
  // Complex64 is not supported.
  'int8': Int8Array,
  // JS does not have a Float16Array.
  // TODO(msoulanille): This will require special handling in C++.
  'float16': Float32Array,
  'float64': Float64Array,
  // Complex128 is not supported.
  // TODO(msoulanille): uint64 is not supported yet because BigInt64Array makes
  // TFJS integration more complicated.
  // 'uint64': BigInt64Array,
  // Resource and Variant are not supported.
  'uint32': Uint32Array,
  'uint16': Uint16Array,
  // TODO(msoulanille): This will require special handling in C++.
  'int4': Uint8Array,
  // TODO(msoulanille): This will require special handling in C++.
  'bfloat16': Float32Array,
  */
});
var SUPPORTED_DTYPES = new Set(Object.keys(DTYPE_TO_ARRAY_TYPE));
function typedArrayToDtype(data) {
  if (data instanceof Float32Array) {
    return "float32";
  } else if (data instanceof Int32Array) {
    return "int32";
  }
  throw new Error(
    `Unsupported typed array type ${data.constructor.name}.`
  );
}

// src/global_litert.ts
var LiteRtNotLoadedError = class extends Error {
  constructor() {
    super(
      "LiteRT is not initialized yet. Please call loadLiteRt() and wait for its promise to resolve to load the LiteRT WASM module."
    );
  }
};
var globalLiteRt = void 0;
var globalLiteRtPromise = void 0;
function getGlobalLiteRt() {
  if (!globalLiteRt) {
    throw new LiteRtNotLoadedError();
  }
  return globalLiteRt;
}
function hasGlobalLiteRt() {
  return Boolean(globalLiteRt);
}
function setGlobalLiteRt(liteRt) {
  globalLiteRt = liteRt;
}
function getGlobalLiteRtPromise() {
  return globalLiteRtPromise;
}
function hasGlobalLiteRtPromise() {
  return Boolean(globalLiteRtPromise);
}
function setGlobalLiteRtPromise(promise) {
  globalLiteRtPromise = promise;
}

// src/tensor.ts
function isTensorReferenceData(data) {
  const maybeData = data;
  return maybeData !== void 0 && typeof maybeData === "object" && typeof maybeData.type === "object" && typeof maybeData.accelerator === "string" && typeof maybeData.reference === "object";
}
var Tensor = class _Tensor {
  // This contains properties of TensorWrapper but organized in a more
  // JS-friendly way. Some properties may be missing, such as when the user
  // creates their own Tensor.
  //
  // Additionally, instances of this interface are not associated with a
  // specific TfLite Interpreter.
  static copyFunctions = {};
  tensorReferenceData;
  deletedInternal = false;
  constructor(dataOrTypedArray, shape) {
    if (isTensorReferenceData(dataOrTypedArray)) {
      this.tensorReferenceData = dataOrTypedArray;
    } else {
      this.tensorReferenceData = {
        type: {
          dtype: typedArrayToDtype(dataOrTypedArray),
          layout: {
            dimensions: shape ?? [dataOrTypedArray.length]
          }
        },
        accelerator: "wasm",
        reference: typedArrayToCpuTensorReference(dataOrTypedArray)
      };
    }
  }
  /**
   * Returns the datatype of the tensor.
   */
  get type() {
    return this.tensorReferenceData.type;
  }
  /**
   * Returns the accelerator the tensor is stored on.
   */
  get accelerator() {
    return this.tensorReferenceData.accelerator;
  }
  /**
   * Returns the internal reference to the tensor data.
   *
   * Users should not rely on this call, and should use `toTypedArray` instead
   * if they are trying to view Tensor data.
   */
  get reference() {
    return this.tensorReferenceData.reference;
  }
  static fromTypedArray(data, shape) {
    return new _Tensor(data, shape);
  }
  /**
   * Returns the data of the tensor as a TypedArray.
   *
   * The returned TypedArray is a copy of the data, and this method does not
   * delete the original tensor.
   * @throws An error if the tensor is not on Wasm.
   */
  toTypedArray() {
    if (this.accelerator !== "wasm") {
      throw new Error(
        "Tensor must be on Wasm to be converted to a TypedArray."
      );
    }
    const typedArrayConstructor = DTYPE_TO_ARRAY_TYPE[this.type.dtype];
    const cpuTensorReference = this.reference;
    const data = cpuTensorReference.data();
    return new typedArrayConstructor(
      // Cast is needed to avoid 'SharedArrayBuffer' in the type.
      data.buffer,
      data.byteOffset,
      data.length / typedArrayConstructor.BYTES_PER_ELEMENT
    ).slice();
  }
  /**
   * Copies the tensor to the given accelerator.
   *
   * @param accelerator The accelerator to copy to.
   * @return A promise that resolves to the copied tensor.
   */
  async copyTo(accelerator) {
    const copyFunctions = _Tensor.copyFunctions[this.accelerator];
    if (!copyFunctions) {
      throw new Error(
        `Accelerator ${this.accelerator} does not support copying`
      );
    }
    const copyFunctionSet = copyFunctions[accelerator];
    if (!copyFunctionSet || !copyFunctionSet.copyTo) {
      const supportedCopyDestinations = Object.entries(copyFunctions).filter(([key, value]) => value.copyTo).map(([key, value]) => key);
      throw new Error(`Accelerator ${this.accelerator} does not support copying to ${accelerator}. It supports copying to the following accelerators: [${supportedCopyDestinations.join(", ")}].`);
    }
    return copyFunctionSet.copyTo(this);
  }
  /**
   * Moves the tensor to the given accelerator, deleting the original.
   *
   * @param accelerator The accelerator to move to.
   * @return A promise that resolves to the moved tensor.
   */
  async moveTo(accelerator) {
    const copyFunctions = _Tensor.copyFunctions[this.accelerator];
    if (!copyFunctions) {
      throw new Error(
        `Accelerator ${this.accelerator} does not support moving`
      );
    }
    const copyFunctionSet = copyFunctions[accelerator];
    if (!copyFunctionSet || !copyFunctionSet.moveTo) {
      const supportedMoveDestinations = Object.entries(copyFunctions).filter(([key, value]) => value.moveTo).map(([key, value]) => key);
      throw new Error(`Accelerator ${this.accelerator} does not support moving to ${accelerator}. It supports moving to the following accelerators: [${supportedMoveDestinations.join(", ")}].`);
    }
    return copyFunctionSet.moveTo(this);
  }
  get deleted() {
    return this.deletedInternal;
  }
  delete() {
    this.tensorReferenceData.reference.delete?.();
    this.deletedInternal = true;
  }
};
var TensorTypeError = class extends Error {
  constructor(name, index, expected, actual) {
    super(`Input tensor for ${name} at position ${index} has type ${actual}, but signature expects ${expected}.`);
  }
};
var TensorShapeError = class extends Error {
  constructor(name, expected, actual) {
    const expectedShapeString = `[${expected.join(", ")}]`;
    const actualShapeString = `[${actual.join(", ")}]`;
    super(
      `Input tensor for ${name} has shape ${actualShapeString}, but signature expects ${expectedShapeString}.`
    );
  }
};
function typedArrayToCpuTensorReference(data) {
  const globalLiteRt2 = getGlobalLiteRt();
  const arrayType = data.constructor;
  const cpuTensor = new globalLiteRt2.liteRtWasm.CpuTensor(
    data.length * arrayType.BYTES_PER_ELEMENT
  );
  const cpuTensorUint8Array = cpuTensor.data();
  const cpuTensorArray = new arrayType(
    // Cast is needed to avoid 'SharedArrayBuffer' in the type.
    cpuTensorUint8Array.buffer,
    cpuTensorUint8Array.byteOffset,
    data.length
  );
  cpuTensorArray.set(data);
  return cpuTensor;
}

// src/signature_runner.ts
var SignatureRunner = class {
  constructor(signatureRunnerWrapper) {
    this.signatureRunnerWrapper = signatureRunnerWrapper;
    this.inputTensorsVector = this.signatureRunnerWrapper.inputs();
    for (let i = 0; i < this.inputTensorsVector.size(); ++i) {
      const tensor = this.inputTensorsVector.get(i);
      this.inputTensors.set(tensor.name(), tensor);
    }
    this.outputTensorsVector = this.signatureRunnerWrapper.outputs();
    for (let i = 0; i < this.outputTensorsVector.size(); ++i) {
      const tensor = this.outputTensorsVector.get(i);
      this.outputTensors.set(tensor.name(), tensor);
    }
  }
  inputTensors = /* @__PURE__ */ new Map();
  inputTensorsVector;
  outputTensors = /* @__PURE__ */ new Map();
  outputTensorsVector;
  deleted = false;
  checkTypes(inputs) {
    const inputTensorsList = [...this.inputTensors.values()];
    for (let i = 0; i < inputTensorsList.length; ++i) {
      const tensorWrapper = inputTensorsList[i];
      const tensor = inputs[i];
      const expectedDType = tensorWrapper.type();
      if (expectedDType !== tensor.type.dtype) {
        throw new TensorTypeError(
          tensorWrapper.name(),
          i,
          expectedDType,
          tensor.type.dtype
        );
      }
    }
  }
  run(input) {
    if (this.deleted) {
      throw new Error("Signature has been deleted. Please reload the model.");
    }
    let inputArray;
    let shouldReturnArray = true;
    if (Array.isArray(input)) {
      if (input.length !== this.inputTensors.size) {
        throw new Error(
          `run() called with ${input.length} inputs, but signature expects ${this.inputTensors.size} inputs`
        );
      }
      inputArray = input;
    } else if (input instanceof Tensor) {
      if (this.inputTensors.size !== 1) {
        throw new Error(
          `run() called with a single tensor, but signature expects ${this.inputTensors.size} inputs`
        );
      }
      inputArray = [input];
    } else {
      shouldReturnArray = false;
      inputArray = [];
      for (const name of this.inputTensors.keys()) {
        const tensor = input[name];
        if (!tensor) {
          throw new Error(`Expected input tensor with name '${name}', but none was provided.`);
        }
        inputArray.push(tensor);
      }
    }
    this.checkTypes(inputArray);
    const outputArray = this.runWithArray(inputArray);
    if (shouldReturnArray) {
      return outputArray;
    }
    const output = {};
    const names = [...this.outputTensors.keys()];
    for (let i = 0; i < names.length; i++) {
      output[names[i]] = outputArray[i];
    }
    return output;
  }
  pushErrorScopes() {
  }
  popErrorScopes(callsite) {
  }
  /**
   * Runs the default signature of the model with the given input tensors and
   * returns the outputs.
   */
  runWithArray(input) {
    const tensors = this.signatureRunnerWrapper.makeTensorVector();
    for (const tensor of input) {
      tensors.push_back(tensor.reference);
    }
    this.pushErrorScopes();
    this.signatureRunnerWrapper.copyInputs(tensors);
    this.popErrorScopes("copyInputs");
    tensors.delete();
    this.pushErrorScopes();
    this.signatureRunnerWrapper.invoke();
    this.popErrorScopes("invoke");
    this.pushErrorScopes();
    const outputTensorReferences = this.signatureRunnerWrapper.copyOutputs();
    this.popErrorScopes("copyOutputs");
    const output = [];
    for (let i = 0; i < this.outputTensorsVector.size(); ++i) {
      const tensorWrapper = this.outputTensorsVector.get(i);
      const tensorReference = outputTensorReferences.get(i);
      output.push(new Tensor({
        type: {
          dtype: tensorWrapper.type(),
          layout: { dimensions: tensorWrapper.shape() }
        },
        accelerator: tensorWrapper.accelerator(),
        reference: tensorReference
      }));
      tensorWrapper.delete();
    }
    outputTensorReferences.delete();
    return output;
  }
  /**
   * Get details about each input tensor.
   */
  getInputDetails() {
    return getTensorMapDetails(this.inputTensors);
  }
  /**
   * Get details about each output tensor.
   */
  getOutputDetails() {
    return getTensorMapDetails(this.outputTensors);
  }
  delete() {
    if (this.deleted) {
      return;
    }
    for (const tensor of this.inputTensors.values()) {
      tensor.delete();
    }
    this.inputTensors.clear();
    this.inputTensorsVector.delete();
    for (const tensor of this.outputTensors.values()) {
      tensor.delete();
    }
    this.outputTensors.clear();
    this.outputTensorsVector.delete();
    this.deleted = true;
  }
};
function getTensorMapDetails(tensors) {
  return [...tensors.entries()].map(
    ([name, tensor], index) => ({ name, index, shape: tensor.shape(), dtype: tensor.type() })
  );
}

// src/cpu_signature_runner.ts
var CpuSignatureRunner = class extends SignatureRunner {
  accelerator = "wasm";
  constructor(signatureRunnerWrapper) {
    super(signatureRunnerWrapper);
  }
  /**
   * Throws an error if the input tensors have different shapes than the
   * signature.
   *
   * Note that this may be overrestrictive since it doesn't account for
   * automatically expanding / contracting dimensions (e.g. [1, 1, 224, 224] vs
   * [224, 224]).
   */
  checkShapes(input) {
    let i = 0;
    for (const tensorWrapper of this.inputTensors.values()) {
      const tensor = input[i++];
      const shape = tensor.type.layout.dimensions;
      const expectedShape = tensorWrapper.shape();
      if (expectedShape.length !== shape.length) {
        throw new TensorShapeError(tensorWrapper.name(), expectedShape, shape);
      }
      for (let j = 0; j < shape.length; ++j) {
        if (shape[j] !== expectedShape[j]) {
          throw new TensorShapeError(
            tensorWrapper.name(),
            expectedShape,
            shape
          );
        }
      }
    }
  }
  runWithArray(input) {
    this.checkShapes(input);
    return super.runWithArray(input);
  }
};

// src/gpu_utils.ts
function isWebGPUSupported() {
  return !!(typeof globalThis !== "undefined" && globalThis.navigator && globalThis.navigator.gpu);
}
var ERROR_SCOPES = ["internal", "out-of-memory", "validation"];
function pushErrorScopes(device) {
  for (const scopeType of ERROR_SCOPES) {
    device.pushErrorScope(scopeType);
  }
}
function popErrorScopes(device, callsite, reportError) {
  for (let i = 0; i < ERROR_SCOPES.length; ++i) {
    device.popErrorScope().then((error) => {
      if (error) {
        reportError(error, callsite);
      }
    });
  }
}
function getBhwcShapeFromInputShape(shape) {
  const shape4d = [1, 1, 1, 1];
  switch (shape.length) {
    case 1:
      shape4d[3] = shape[0];
      break;
    case 2:
      shape4d[3] = shape[1];
      shape4d[2] = shape[0];
      break;
    case 3:
      shape4d[3] = shape[2];
      shape4d[2] = shape[1];
      shape4d[1] = shape[0];
      break;
    case 4:
      shape4d[3] = shape[3];
      shape4d[2] = shape[2];
      shape4d[1] = shape[1];
      shape4d[0] = shape[0];
      break;
    default:
      throw new Error(
        "Only 1D~4D tensors are supported, but got shape: " + shape.toString() + "."
      );
  }
  return shape4d;
}
async function gpuTensorToCpuTensor(gpuTensor) {
  const device = await getGlobalLiteRt().getWebGpuDevice();
  const converter = getGlobalLiteRt().getConverterFactory().makeConverterToTfjs(
    gpuTensor.reference
  );
  const buffer = converter.convertToTfjs(gpuTensor.reference);
  const stagingBuffer = device.createBuffer({
    size: buffer.size,
    usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    mappedAtCreation: false
  });
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(buffer, 0, stagingBuffer, 0, buffer.size);
  device.queue.submit([commandEncoder.finish()]);
  await stagingBuffer.mapAsync(GPUMapMode.READ);
  const mappedBuffer = stagingBuffer.getMappedRange();
  const mappedArray = new Uint8Array(mappedBuffer);
  const cpuTensorConstructor = getGlobalLiteRt().liteRtWasm.CpuTensor;
  const cpuTensorReference = new cpuTensorConstructor(mappedArray.byteLength);
  cpuTensorReference.data().set(mappedArray);
  stagingBuffer.unmap();
  stagingBuffer.destroy();
  return new Tensor({
    type: gpuTensor.type,
    accelerator: "wasm",
    reference: cpuTensorReference
  });
}
async function cpuTensorToGpuTensor(cpuTensor) {
  const device = await getGlobalLiteRt().getWebGpuDevice();
  const cpuTensorData = cpuTensor.reference.data();
  const typedArrayConstructor = DTYPE_TO_ARRAY_TYPE[cpuTensor.type.dtype];
  const typedArray = new typedArrayConstructor(
    // Cast is needed to avoid 'SharedArrayBuffer' in the type.
    cpuTensorData.buffer,
    cpuTensorData.byteOffset,
    cpuTensorData.length
  );
  const stagingBuffer = device.createBuffer({
    size: typedArray.byteLength,
    usage: GPUBufferUsage.MAP_WRITE | GPUBufferUsage.COPY_SRC,
    mappedAtCreation: true
  });
  const mappedBuffer = await stagingBuffer.getMappedRange();
  if (typedArray instanceof Float32Array) {
    const mappedArray = new Float32Array(mappedBuffer);
    mappedArray.set(typedArray);
  } else if (typedArray instanceof Int32Array) {
    const mappedArray = new Int32Array(mappedBuffer);
    mappedArray.set(typedArray);
  } else {
    throw new Error(
      "Unsupported typed array type: " + typedArray.constructor.name
    );
  }
  stagingBuffer.unmap();
  const tempBuffer = device.createBuffer({
    size: stagingBuffer.size,
    usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.STORAGE
  });
  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(
    stagingBuffer,
    0,
    tempBuffer,
    0,
    stagingBuffer.size
  );
  device.queue.submit([commandEncoder.finish()]);
  stagingBuffer.destroy();
  const mlDriftShape = getBhwcShapeFromInputShape(cpuTensor.type.layout.dimensions);
  const converter = getGlobalLiteRt().getConverterFactory().makeConverterFromTfjs(
    cpuTensor.type.dtype,
    ...mlDriftShape
  );
  const gpuTensorReference = converter.convertFromTfjs(tempBuffer);
  tempBuffer.destroy();
  return new Tensor({
    type: cpuTensor.type,
    accelerator: "webgpu",
    reference: gpuTensorReference
  });
}

// src/memoize.ts
var TreeMap = class _TreeMap extends Map {
  val;
  constructor() {
    super();
  }
  getMap(key, createIfMissing = false) {
    let map = this;
    for (let i = 0; i < key.length; i++) {
      if (!map.has(key[i])) {
        if (!createIfMissing) {
          return void 0;
        }
        map.set(key[i], new _TreeMap());
      }
      map = map.get(key[i]);
    }
    return map;
  }
  getPath(key) {
    return this.getMap(key)?.val;
  }
  hasPath(key) {
    return this.getMap(key) !== void 0;
  }
  setPath(key, val) {
    const map = this.getMap(
      key,
      /* createIfMissing= */
      true
    );
    map.val = val;
  }
};
function memoize(f, getKey = (x) => x) {
  const cache = new TreeMap();
  return (...args) => {
    const key = getKey(args);
    if (!cache.hasPath(key)) {
      cache.setPath(key, f(...args));
    }
    return cache.getPath(key);
  };
}

// src/gpu_conversion.ts
var InputConverter = class {
  constructor(converter, wasm, gpuErrorReporter) {
    this.converter = converter;
    this.wasm = wasm;
    this.gpuErrorReporter = gpuErrorReporter;
  }
  convertFromTfjs(buffer) {
    pushErrorScopes(this.wasm.preinitializedWebGPUDevice);
    const bufferPtr = this.wasm.WebGPU.importJsBuffer(buffer);
    const tensor = this.converter.convertFromTfjs(bufferPtr);
    popErrorScopes(
      this.wasm.preinitializedWebGPUDevice,
      "convertFromTfjs",
      this.gpuErrorReporter.val
    );
    return tensor;
  }
  delete() {
    this.converter.delete();
  }
};
var OutputConverter = class {
  constructor(converter, wasm, gpuErrorReporter) {
    this.converter = converter;
    this.wasm = wasm;
    this.gpuErrorReporter = gpuErrorReporter;
  }
  convertToTfjs(tensor) {
    pushErrorScopes(this.wasm.preinitializedWebGPUDevice);
    const bufferPtr = this.converter.convertToTfjs(tensor);
    const buffer = this.wasm.WebGPU.getJsObject(bufferPtr);
    popErrorScopes(
      this.wasm.preinitializedWebGPUDevice,
      "convertToTfjs",
      this.gpuErrorReporter.val
    );
    return buffer;
  }
  delete() {
    this.converter.delete();
  }
};
var ConverterFactory = class {
  constructor(wasm, gpuErrorReporter) {
    this.wasm = wasm;
    this.gpuErrorReporter = gpuErrorReporter;
  }
  /**
   * Returns true if this ConverterFactory uses the same WebGPU device as the
   * one passed in.
   */
  isWebGpuDeviceCompatible(device) {
    return device === this.wasm.preinitializedWebGPUDevice;
  }
  /**
   * Returns an InputConverter for quickly converting WebGPU buffers in TF.js
   * tensor format into the corresponding LiteRT Tensors. Each InputConverter is
   * created for a given type and [B,H,W,C] shape, so the converter can be
   * reused, but only for tensors of the same type and shape.
   */
  makeConverterFromTfjs = memoize(this.makeConverterFromTfjsInternal.bind(this));
  makeConverterFromTfjsInternal(type, b, h, w, c) {
    pushErrorScopes(this.wasm.preinitializedWebGPUDevice);
    const nativeConverter = this.wasm.makeConverterFromTfjs(type, b, h, w, c);
    popErrorScopes(
      this.wasm.preinitializedWebGPUDevice,
      "makeConverterFromTfjs",
      this.gpuErrorReporter.val
    );
    return new InputConverter(
      nativeConverter,
      this.wasm,
      this.gpuErrorReporter
    );
  }
  /**
   * Returns an OutputConverter for quickly converting LiteRT Tensors into the
   * the corresponding WebGPU buffer in TF.js tensor format. Each
   * OutputConverter is created to match the specifications of the given Tensor
   * (type and [B,H,W,C] shape), so the converter can be reused, but only for
   * Tensors of the same type and shape.
   */
  makeConverterToTfjs = memoize(
    this.makeConverterToTfjsInternal.bind(this),
    ([opaqueReference]) => {
      const driftTensor = opaqueReference;
      return driftTensor.getCacheKey();
    }
  );
  makeConverterToTfjsInternal(tensorReference) {
    pushErrorScopes(this.wasm.preinitializedWebGPUDevice);
    const nativeConverter = this.wasm.makeConverterToTfjs(tensorReference);
    popErrorScopes(
      this.wasm.preinitializedWebGPUDevice,
      "makeConverterToTfjs",
      this.gpuErrorReporter.val
    );
    return new OutputConverter(
      nativeConverter,
      this.wasm,
      this.gpuErrorReporter
    );
  }
};

// src/gpu_signature_runner.ts
var GpuSignatureRunner = class extends SignatureRunner {
  constructor(signatureRunnerWrapper, device, gpuErrorReporter) {
    super(signatureRunnerWrapper);
    this.device = device;
    this.gpuErrorReporter = gpuErrorReporter;
  }
  accelerator = "webgpu";
  pushErrorScopes() {
    pushErrorScopes(this.device);
  }
  popErrorScopes(callsite) {
    popErrorScopes(this.device, callsite, this.gpuErrorReporter.val);
  }
};

// src/litert_web.ts
var DESIRED_WEBGPU_FEATURES = [
  "shader-f16",
  "subgroups",
  // In origin trial
  "subgroups-f16"
  // In origin trial
];
function setErrorReporter(errorReporter) {
  getGlobalLiteRt().setErrorReporter(errorReporter);
}
function setGpuErrorReporter(errorReporter) {
  getGlobalLiteRt().setGpuErrorReporter(errorReporter);
}
function setWebGpuDevice(device, adapterInfo) {
  getGlobalLiteRt().setWebGpuDevice(device, adapterInfo);
}
function getWebGpuDevice() {
  return getGlobalLiteRt().getWebGpuDevice();
}
function getAdapterInfo() {
  return getGlobalLiteRt().getAdapterInfo();
}
function loadAndCompile(model, compileOptions) {
  return getGlobalLiteRt().loadAndCompile(model, compileOptions);
}
var LiteRt = class _LiteRt {
  liteRtWasm;
  device;
  // Boxed so it can be passed as a reference to the signatures and updated
  // later.
  gpuErrorReporter = {
    val: (error, callsite) => {
      console.error("GPU error:", error, "at:", callsite);
    }
  };
  loadAndCompileWebGpuWasCalled = false;
  loadedModels = /* @__PURE__ */ new Set();
  converterFactory;
  constructor(wasmModule) {
    this.liteRtWasm = wasmModule;
    if (!this.liteRtWasm.loadAndCompileWebGpu) {
      throw new Error("loadAndCompileWebGpu is not defined.");
    }
    this.liteRtWasm.setupLogging();
  }
  pushErrorScopes() {
    if (!this.device) {
      throw new Error("No GPU device provided.");
    }
    pushErrorScopes(this.device);
  }
  popErrorScopes(callsite) {
    if (!this.device) {
      throw new Error("No GPU device provided.");
    }
    popErrorScopes(this.device, callsite, this.gpuErrorReporter.val);
  }
  static async urlToUint8Array(url) {
    const response = await fetch(url);
    return new Uint8Array(await response.arrayBuffer());
  }
  static async readableStreamToUint8Array(reader) {
    let byteOffset = 0;
    let array = new Uint8Array(
      1024
      /* arbitrary starting size */
    );
    const MAX_ARRAY_SIZE = 2e9;
    while (true) {
      const { done, value } = await reader.read();
      if (value) {
        if (array.byteLength < byteOffset + value.byteLength) {
          if (byteOffset + value.byteLength > MAX_ARRAY_SIZE) {
            throw new Error(`Model is too large (> ${MAX_ARRAY_SIZE} bytes`);
          }
          const newArray = new Uint8Array(Math.min(
            MAX_ARRAY_SIZE,
            Math.max(array.byteLength, value.byteLength) * 2
          ));
          newArray.set(array);
          array = newArray;
        }
        array.set(value, byteOffset);
        byteOffset += value.byteLength;
      }
      if (done) {
        break;
      }
    }
    return array.slice(0, byteOffset);
  }
  /**
   * Initialize the WebGPU device for LiteRT.
   */
  async initializeDefaultWebGpuDevice() {
    if (this.device) {
      console.warn("WebGPU device is already initialized.");
      return;
    }
    if (!isWebGPUSupported()) {
      throw new Error("This browser does not support WebGPU.");
    }
    const adapterDescriptor = {
      powerPreference: "high-performance"
    };
    const adapter = await navigator.gpu.requestAdapter(adapterDescriptor);
    if (!adapter) {
      throw new Error("No GPU adapter found.");
    }
    const adapterInfo = adapter.info;
    const requiredLimits = {
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxStorageBuffersPerShaderStage: adapter.limits.maxStorageBuffersPerShaderStage,
      maxTextureDimension2D: adapter.limits.maxTextureDimension2D
    };
    const requiredFeatures = [];
    for (const feature of DESIRED_WEBGPU_FEATURES) {
      if (adapter.features.has(feature)) {
        requiredFeatures.push(feature);
      }
    }
    const device = await adapter.requestDevice({
      requiredFeatures,
      requiredLimits
    });
    this.setWebGpuDevice(device, adapterInfo);
  }
  /**
   * Set the error reporter for LiteRt.
   */
  setErrorReporter(errorReporter) {
    this.liteRtWasm.setErrorReporter(errorReporter);
  }
  /**
   * Set the WebGPU error reporter for LiteRt.
   */
  setGpuErrorReporter(errorReporter) {
    this.gpuErrorReporter.val = errorReporter;
  }
  /**
   * Set the WebGPU device and adapter info for LiteRT.
   */
  // TODO: Remove adapterInfo from the api, as the latest GPUDevice type should
  // have adapterInfo.
  setWebGpuDevice(device, adapterInfo) {
    if (this.loadAndCompileWebGpuWasCalled) {
      throw new Error(
        "The WebGPU device cannot be set after loading a WebGPU model."
      );
    }
    this.device = device;
    if (!this.device.adapterInfo) {
      if (!adapterInfo) {
        throw new Error(
          "The device does not have adapter info, so adapterInfo must be provided."
        );
      }
      this.device.adapterInfo = adapterInfo;
    }
    this.liteRtWasm.preinitializedWebGPUDevice = this.device;
  }
  /**
   * Get the WebGPU device that LiteRT is using. If the device is not set,
   * initialize it.
   */
  async getWebGpuDevice() {
    if (!this.device) {
      await this.initializeDefaultWebGpuDevice();
    }
    return this.device;
  }
  /**
   * Get the WebGPU adapter info that LiteRT is using. If the WebGPU device is
   * not set, initialize it.
   */
  async getAdapterInfo() {
    if (!this.device) {
      await this.initializeDefaultWebGpuDevice();
    }
    return this.device.adapterInfo;
  }
  /**
   * Loads a LiteRt model.
   *
   * @param model The model data. This can be a string (the model url), a URL
   *     object, a Uint8Array (the model bytes), or a
   *     ReadableStreamDefaultReader (for streaming model loading).
   * @param compileOptions The options for compiling the model. This includes
   *     the accelerator to use ('webgpu' or 'wasm') and the WebGPU device
   *     (for direct GPU model inputs / outputs).
   * @returns A promise that resolves to the CompiledModel.
   */
  async loadAndCompile(model, compileOptions) {
    let modelData;
    if (typeof model === "string" || model instanceof URL) {
      modelData = await _LiteRt.urlToUint8Array(model);
    } else if (model instanceof Uint8Array) {
      modelData = model;
    } else if (model instanceof ReadableStreamDefaultReader) {
      modelData = await _LiteRt.readableStreamToUint8Array(model);
    } else {
      throw new Error("Unsupported model type.");
    }
    const ptr = this.liteRtWasm._malloc(modelData.byteLength);
    this.liteRtWasm.HEAPU8.set(modelData, ptr);
    let compiledModel;
    const onDelete = () => {
      this.liteRtWasm._free(ptr);
      this.loadedModels.delete(compiledModel);
    };
    if (compileOptions.accelerator === "webgpu") {
      if (!this.liteRtWasm.preinitializedWebGPUDevice) {
        await this.initializeDefaultWebGpuDevice();
      }
      this.pushErrorScopes();
      this.loadAndCompileWebGpuWasCalled = true;
      const liteRtInterpreter = this.liteRtWasm.loadAndCompileWebGpu(ptr, modelData.byteLength);
      this.popErrorScopes("loadAndCompile");
      compiledModel = new CompiledModel(liteRtInterpreter, (signatureRunnerWrapper) => {
        if (!this.device) {
          throw new Error("No GPU device provided.");
        }
        return new GpuSignatureRunner(
          signatureRunnerWrapper,
          this.device,
          this.gpuErrorReporter
        );
      }, onDelete);
    } else {
      const liteRtInterpreter = this.liteRtWasm.loadAndCompileCpu(ptr, modelData.byteLength);
      compiledModel = new CompiledModel(liteRtInterpreter, (signatureRunnerWrapper) => {
        return new CpuSignatureRunner(signatureRunnerWrapper);
      }, onDelete);
    }
    this.loadedModels.add(compiledModel);
    return compiledModel;
  }
  /**
   * Gets or creates a ConverterFactory for our tensor converters.
   */
  getConverterFactory() {
    if (!this.converterFactory) {
      this.converterFactory = new ConverterFactory(this.liteRtWasm, this.gpuErrorReporter);
    }
    return this.converterFactory;
  }
  /**
   * Delete the LiteRt wasm module and all loaded models.
   */
  delete() {
    for (const model of this.loadedModels) {
      model.delete();
    }
  }
};

// src/load.ts
import { createWasmLib } from "./wasm-utils.js";

// src/url_path_utils.ts
function pathToString(path) {
  return path;
}
function appendPathSegment(path, segment) {
  if (!path) return segment;
  if (!segment) return path;
  const pathWithSlash = path.endsWith("/") ? path : path + "/";
  const segmentWithoutSlash = segment.startsWith("/") ? segment.substring(1) : segment;
  return pathWithSlash + segmentWithoutSlash;
}

// src/wasm_feature_detect.ts
var WASM_RELAXED_SIMD_CHECK = new Uint8Array([
  0,
  97,
  115,
  109,
  1,
  0,
  0,
  0,
  1,
  5,
  1,
  96,
  0,
  1,
  123,
  3,
  2,
  1,
  0,
  10,
  15,
  1,
  13,
  0,
  65,
  1,
  253,
  15,
  65,
  2,
  253,
  15,
  253,
  128,
  2,
  11
]);
var WASM_THREADS_CHECK = new Uint8Array([
  0,
  97,
  115,
  109,
  1,
  0,
  0,
  0,
  1,
  4,
  1,
  96,
  0,
  0,
  3,
  2,
  1,
  0,
  5,
  4,
  1,
  3,
  1,
  1,
  10,
  11,
  1,
  9,
  0,
  65,
  0,
  254,
  16,
  2,
  0,
  26,
  11
]);
var WASM_FEATURE_VALUES = {
  "relaxedSimd": void 0,
  "threads": void 0
};
async function tryWasm(wasm) {
  try {
    await WebAssembly.instantiate(wasm);
    return { supported: true };
  } catch (e) {
    return { supported: false, error: e };
  }
}
var WASM_FEATURE_CHECKS = {
  "relaxedSimd": () => {
    if (WASM_FEATURE_VALUES.relaxedSimd === void 0) {
      WASM_FEATURE_VALUES.relaxedSimd = tryWasm(WASM_RELAXED_SIMD_CHECK);
    }
    return WASM_FEATURE_VALUES.relaxedSimd;
  },
  "threads": () => {
    if (WASM_FEATURE_VALUES.threads === void 0) {
      try {
        if (typeof MessageChannel !== "undefined") {
          new MessageChannel().port1.postMessage(new SharedArrayBuffer(1));
        }
        WASM_FEATURE_VALUES.threads = tryWasm(WASM_THREADS_CHECK);
      } catch (e) {
        WASM_FEATURE_VALUES.threads = Promise.resolve({ supported: false, error: e });
      }
    }
    return WASM_FEATURE_VALUES.threads;
  }
};
async function supportsFeature(feature) {
  const check = WASM_FEATURE_CHECKS[feature]?.();
  if (!check) {
    throw new Error(`Unknown feature: ${feature}`);
  }
  return (await check).supported;
}
async function throwIfFeatureNotSupported(feature) {
  const check = WASM_FEATURE_CHECKS[feature]?.();
  if (!check) {
    throw new Error(`Unknown feature: ${feature}`);
  }
  const result = await check;
  if (!result.supported) {
    throw result.error;
  }
}

// src/load.ts
var WASM_JS_FILE_NAME = "litert_wasm_internal.js";
var WASM_JS_COMPAT_FILE_NAME = "litert_wasm_compat_internal.js";
var WASM_JS_THREADED_FILE_NAME = "litert_wasm_threaded_internal.js";
async function load(path, options) {
  const pathString = pathToString(path);
  const isFullFilePath = pathString.endsWith(".wasm") || pathString.endsWith(".js");
  const relaxedSimd = await supportsFeature("relaxedSimd");
  if (options?.threads) {
    if (isFullFilePath) {
      console.warn(
        `The \`threads\` option was specified, but the wasm path ${pathString} is a full file path. Whether threads are available or not will depend on the loaded file. To allow LiteRT.js to load the threaded wasm file, use a directory path instead of a full file path.`
      );
    }
    if (!relaxedSimd) {
      throw new Error(
        "Threads are only supported with relaxed SIMD, and the current browser does not support relaxed SIMD."
      );
    }
    await throwIfFeatureNotSupported("threads");
  }
  let fileName = WASM_JS_COMPAT_FILE_NAME;
  if (relaxedSimd) {
    if (options?.threads) {
      fileName = WASM_JS_THREADED_FILE_NAME;
    } else {
      fileName = WASM_JS_FILE_NAME;
    }
  }
  let jsFilePath = path;
  if (pathString.endsWith(".wasm")) {
    throw new Error(
      "Please load the `.js` file corresponding to the `.wasm` file, or load the directory containing it."
    );
  } else if (!pathString.endsWith(".js")) {
    jsFilePath = appendPathSegment(path, fileName);
  }
  return createWasmLib(LiteRt, jsFilePath);
}

// src/load_litert.ts
function loadLiteRt(path, options) {
  if (hasGlobalLiteRtPromise()) {
    throw new Error("LiteRT is already loading / loaded.");
  }
  setGlobalLiteRtPromise(load(path, options).then((liteRt) => {
    setGlobalLiteRt(liteRt);
    return liteRt;
  }).catch((error) => {
    setGlobalLiteRtPromise(void 0);
    throw error;
  }));
  return getGlobalLiteRtPromise();
}
function unloadLiteRt() {
  if (hasGlobalLiteRtPromise() && !hasGlobalLiteRt()) {
    throw new Error(
      "LiteRT is loading and can not be unloaded or canceled until it is finished loading."
    );
  }
  if (hasGlobalLiteRt()) {
    getGlobalLiteRt().delete();
    setGlobalLiteRt(void 0);
  }
  setGlobalLiteRtPromise(void 0);
}

// src/tensor_copy_functions.ts
function registerCopyFunctions() {
  Tensor.copyFunctions["wasm"] = {
    "webgpu": {
      copyTo: cpuTensorToGpuTensor,
      moveTo: async (tensor) => {
        const gpuTensor = await cpuTensorToGpuTensor(tensor);
        tensor.delete();
        return gpuTensor;
      }
    }
  };
  Tensor.copyFunctions["webgpu"] = {
    "wasm": {
      copyTo: gpuTensorToCpuTensor,
      moveTo: async (tensor) => {
        const cpuTensor = await gpuTensorToCpuTensor(tensor);
        tensor.delete();
        return cpuTensor;
      }
    }
  };
}

// src/index.ts
registerCopyFunctions();
export {
  ACCELERATORS,
  CompiledModel,
  ConverterFactory,
  CpuSignatureRunner,
  DTYPE_TO_ARRAY_TYPE,
  GpuSignatureRunner,
  InputConverter,
  LiteRt,
  LiteRtNotLoadedError,
  OutputConverter,
  SUPPORTED_DTYPES,
  SignatureRunner,
  Tensor,
  TensorShapeError,
  TensorTypeError,
  getAdapterInfo,
  getGlobalLiteRt,
  getGlobalLiteRtPromise,
  getWebGpuDevice,
  isWebGPUSupported,
  loadAndCompile,
  loadLiteRt,
  setErrorReporter,
  setGpuErrorReporter,
  setWebGpuDevice,
  typedArrayToDtype,
  unloadLiteRt
};
/**
 * @fileoverview A memoization utility for JavaScript.
 *
 * This utility provides a function `memoize` that can be used to memoize
 * functions. A memoized function will only be called once for each unique set
 * of arguments, and the result will be cached and returned on subsequent calls.
 *
 * Example usage:
 *
 * ```typescript
 * const memoizedAdd = memoize((a, b) => a + b);
 * console.log(memoizedAdd(1, 2)); // Output: 3
 * console.log(memoizedAdd(1, 2)); // Output: 3
 * ```
 *
 * In this example, the `memoizedAdd` function will only be called once, even
 * though it is called twice. The result of the first call will be cached and
 * returned on the second call.
 *
 * @license
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
//# sourceMappingURL=index.js.map
