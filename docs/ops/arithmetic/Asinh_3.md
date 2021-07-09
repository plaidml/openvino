## Asinh <a name="Asinh"></a> {#openvino_docs_ops_arithmetic_Asinh_3}

**Versioned name**: *Asinh-3*

**Category**: Arithmetic unary operation

**Short description**: *Asinh* performs element-wise inverse hyperbolic sine operation (arcsinh) on a given input tensor.

**Detailed description**: *Asinh* performs element-wise inverse hyperbolic sine operation on a given input tensor, based on the following mathematical formula:

\f[
a_{i} = asinh(a_{i})
\f]

**Attributes**: *Asinh* operation has no attributes.

**Inputs**

* **1**: A tensor of type *T* and arbitrary shape. **Required.**

**Outputs**

* **1**: The result of element-wise *Asinh* operation. A tensor of type *T* and the same shape as input tensor.

**Types**

* *T*: any numeric type.

**Example**

```xml
<layer ... type="Asinh">
    <input>
        <port id="0">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </input>
    <output>
        <port id="1">
            <dim>256</dim>
            <dim>56</dim>
        </port>
    </output>
</layer>
```
