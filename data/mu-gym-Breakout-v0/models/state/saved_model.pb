�<
��
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape�"serve*2.1.02v2.1.0-rc2-17-ge5bf8de8��3
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
: *
dtype0
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
: *
dtype0
�
res_conv_A_repr_a/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *)
shared_nameres_conv_A_repr_a/kernel
�
,res_conv_A_repr_a/kernel/Read/ReadVariableOpReadVariableOpres_conv_A_repr_a/kernel*&
_output_shapes
:  *
dtype0
�
res_conv_A_repr_a/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameres_conv_A_repr_a/bias
}
*res_conv_A_repr_a/bias/Read/ReadVariableOpReadVariableOpres_conv_A_repr_a/bias*
_output_shapes
: *
dtype0
�
batch_normalization/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_namebatch_normalization/gamma
�
-batch_normalization/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization/gamma*
_output_shapes
: *
dtype0
�
batch_normalization/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *)
shared_namebatch_normalization/beta
�
,batch_normalization/beta/Read/ReadVariableOpReadVariableOpbatch_normalization/beta*
_output_shapes
: *
dtype0
�
batch_normalization/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!batch_normalization/moving_mean
�
3batch_normalization/moving_mean/Read/ReadVariableOpReadVariableOpbatch_normalization/moving_mean*
_output_shapes
: *
dtype0
�
#batch_normalization/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#batch_normalization/moving_variance
�
7batch_normalization/moving_variance/Read/ReadVariableOpReadVariableOp#batch_normalization/moving_variance*
_output_shapes
: *
dtype0
�
res_conv_B_repr_a/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:  *)
shared_nameres_conv_B_repr_a/kernel
�
,res_conv_B_repr_a/kernel/Read/ReadVariableOpReadVariableOpres_conv_B_repr_a/kernel*&
_output_shapes
:  *
dtype0
�
res_conv_B_repr_a/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_nameres_conv_B_repr_a/bias
}
*res_conv_B_repr_a/bias/Read/ReadVariableOpReadVariableOpres_conv_B_repr_a/bias*
_output_shapes
: *
dtype0
�
batch_normalization_1/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namebatch_normalization_1/gamma
�
/batch_normalization_1/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_1/gamma*
_output_shapes
: *
dtype0
�
batch_normalization_1/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namebatch_normalization_1/beta
�
.batch_normalization_1/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_1/beta*
_output_shapes
: *
dtype0
�
!batch_normalization_1/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!batch_normalization_1/moving_mean
�
5batch_normalization_1/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_1/moving_mean*
_output_shapes
: *
dtype0
�
%batch_normalization_1/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape: *6
shared_name'%batch_normalization_1/moving_variance
�
9batch_normalization_1/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_1/moving_variance*
_output_shapes
: *
dtype0
�
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
: @*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
�
res_conv_A_repr_b/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*)
shared_nameres_conv_A_repr_b/kernel
�
,res_conv_A_repr_b/kernel/Read/ReadVariableOpReadVariableOpres_conv_A_repr_b/kernel*&
_output_shapes
:@@*
dtype0
�
res_conv_A_repr_b/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameres_conv_A_repr_b/bias
}
*res_conv_A_repr_b/bias/Read/ReadVariableOpReadVariableOpres_conv_A_repr_b/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_2/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_2/gamma
�
/batch_normalization_2/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_2/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_2/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_2/beta
�
.batch_normalization_2/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_2/beta*
_output_shapes
:@*
dtype0
�
!batch_normalization_2/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_2/moving_mean
�
5batch_normalization_2/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_2/moving_mean*
_output_shapes
:@*
dtype0
�
%batch_normalization_2/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_2/moving_variance
�
9batch_normalization_2/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_2/moving_variance*
_output_shapes
:@*
dtype0
�
res_conv_B_repr_b/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*)
shared_nameres_conv_B_repr_b/kernel
�
,res_conv_B_repr_b/kernel/Read/ReadVariableOpReadVariableOpres_conv_B_repr_b/kernel*&
_output_shapes
:@@*
dtype0
�
res_conv_B_repr_b/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameres_conv_B_repr_b/bias
}
*res_conv_B_repr_b/bias/Read/ReadVariableOpReadVariableOpres_conv_B_repr_b/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_3/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_3/gamma
�
/batch_normalization_3/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_3/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_3/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_3/beta
�
.batch_normalization_3/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_3/beta*
_output_shapes
:@*
dtype0
�
!batch_normalization_3/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_3/moving_mean
�
5batch_normalization_3/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_3/moving_mean*
_output_shapes
:@*
dtype0
�
%batch_normalization_3/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_3/moving_variance
�
9batch_normalization_3/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_3/moving_variance*
_output_shapes
:@*
dtype0
�
res_conv_A_repr_c/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*)
shared_nameres_conv_A_repr_c/kernel
�
,res_conv_A_repr_c/kernel/Read/ReadVariableOpReadVariableOpres_conv_A_repr_c/kernel*&
_output_shapes
:@@*
dtype0
�
res_conv_A_repr_c/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameres_conv_A_repr_c/bias
}
*res_conv_A_repr_c/bias/Read/ReadVariableOpReadVariableOpres_conv_A_repr_c/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_4/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_4/gamma
�
/batch_normalization_4/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_4/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_4/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_4/beta
�
.batch_normalization_4/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_4/beta*
_output_shapes
:@*
dtype0
�
!batch_normalization_4/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_4/moving_mean
�
5batch_normalization_4/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_4/moving_mean*
_output_shapes
:@*
dtype0
�
%batch_normalization_4/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_4/moving_variance
�
9batch_normalization_4/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_4/moving_variance*
_output_shapes
:@*
dtype0
�
res_conv_B_repr_c/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*)
shared_nameres_conv_B_repr_c/kernel
�
,res_conv_B_repr_c/kernel/Read/ReadVariableOpReadVariableOpres_conv_B_repr_c/kernel*&
_output_shapes
:@@*
dtype0
�
res_conv_B_repr_c/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameres_conv_B_repr_c/bias
}
*res_conv_B_repr_c/bias/Read/ReadVariableOpReadVariableOpres_conv_B_repr_c/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_5/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_5/gamma
�
/batch_normalization_5/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_5/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_5/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_5/beta
�
.batch_normalization_5/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_5/beta*
_output_shapes
:@*
dtype0
�
!batch_normalization_5/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_5/moving_mean
�
5batch_normalization_5/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_5/moving_mean*
_output_shapes
:@*
dtype0
�
%batch_normalization_5/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_5/moving_variance
�
9batch_normalization_5/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_5/moving_variance*
_output_shapes
:@*
dtype0
�
res_conv_A_repr_d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*)
shared_nameres_conv_A_repr_d/kernel
�
,res_conv_A_repr_d/kernel/Read/ReadVariableOpReadVariableOpres_conv_A_repr_d/kernel*&
_output_shapes
:@@*
dtype0
�
res_conv_A_repr_d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameres_conv_A_repr_d/bias
}
*res_conv_A_repr_d/bias/Read/ReadVariableOpReadVariableOpres_conv_A_repr_d/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_6/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_6/gamma
�
/batch_normalization_6/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_6/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_6/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_6/beta
�
.batch_normalization_6/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_6/beta*
_output_shapes
:@*
dtype0
�
!batch_normalization_6/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_6/moving_mean
�
5batch_normalization_6/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_6/moving_mean*
_output_shapes
:@*
dtype0
�
%batch_normalization_6/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_6/moving_variance
�
9batch_normalization_6/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_6/moving_variance*
_output_shapes
:@*
dtype0
�
res_conv_B_repr_d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*)
shared_nameres_conv_B_repr_d/kernel
�
,res_conv_B_repr_d/kernel/Read/ReadVariableOpReadVariableOpres_conv_B_repr_d/kernel*&
_output_shapes
:@@*
dtype0
�
res_conv_B_repr_d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*'
shared_nameres_conv_B_repr_d/bias
}
*res_conv_B_repr_d/bias/Read/ReadVariableOpReadVariableOpres_conv_B_repr_d/bias*
_output_shapes
:@*
dtype0
�
batch_normalization_7/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_7/gamma
�
/batch_normalization_7/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_7/gamma*
_output_shapes
:@*
dtype0
�
batch_normalization_7/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*+
shared_namebatch_normalization_7/beta
�
.batch_normalization_7/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_7/beta*
_output_shapes
:@*
dtype0
�
!batch_normalization_7/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*2
shared_name#!batch_normalization_7/moving_mean
�
5batch_normalization_7/moving_mean/Read/ReadVariableOpReadVariableOp!batch_normalization_7/moving_mean*
_output_shapes
:@*
dtype0
�
%batch_normalization_7/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*6
shared_name'%batch_normalization_7/moving_variance
�
9batch_normalization_7/moving_variance/Read/ReadVariableOpReadVariableOp%batch_normalization_7/moving_variance*
_output_shapes
:@*
dtype0
�
repr_board/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*"
shared_namerepr_board/kernel

%repr_board/kernel/Read/ReadVariableOpReadVariableOprepr_board/kernel*&
_output_shapes
:@*
dtype0
v
repr_board/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namerepr_board/bias
o
#repr_board/bias/Read/ReadVariableOpReadVariableOprepr_board/bias*
_output_shapes
:*
dtype0

NoOpNoOp
��
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer_with_weights-1
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer_with_weights-8
layer-16
layer-17
layer_with_weights-9
layer-18
layer-19
layer-20
layer_with_weights-10
layer-21
layer-22
layer_with_weights-11
layer-23
layer_with_weights-12
layer-24
layer-25
layer_with_weights-13
layer-26
layer-27
layer-28
layer_with_weights-14
layer-29
layer-30
 layer_with_weights-15
 layer-31
!layer_with_weights-16
!layer-32
"layer-33
#layer_with_weights-17
#layer-34
$layer-35
%layer_with_weights-18
%layer-36
&trainable_variables
'regularization_losses
(	variables
)	keras_api
*
signatures
 
R
+trainable_variables
,regularization_losses
-	variables
.	keras_api
R
/trainable_variables
0regularization_losses
1	variables
2	keras_api
R
3trainable_variables
4regularization_losses
5	variables
6	keras_api
h

7kernel
8bias
9trainable_variables
:regularization_losses
;	variables
<	keras_api
h

=kernel
>bias
?trainable_variables
@regularization_losses
A	variables
B	keras_api
R
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
�
Gaxis
	Hgamma
Ibeta
Jmoving_mean
Kmoving_variance
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
h

Pkernel
Qbias
Rtrainable_variables
Sregularization_losses
T	variables
U	keras_api
R
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
�
Zaxis
	[gamma
\beta
]moving_mean
^moving_variance
_trainable_variables
`regularization_losses
a	variables
b	keras_api
R
ctrainable_variables
dregularization_losses
e	variables
f	keras_api
h

gkernel
hbias
itrainable_variables
jregularization_losses
k	variables
l	keras_api
h

mkernel
nbias
otrainable_variables
pregularization_losses
q	variables
r	keras_api
R
strainable_variables
tregularization_losses
u	variables
v	keras_api
�
waxis
	xgamma
ybeta
zmoving_mean
{moving_variance
|trainable_variables
}regularization_losses
~	variables
	keras_api
n
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
n
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�trainable_variables
�regularization_losses
�	variables
�	keras_api
n
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
n
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�trainable_variables
�regularization_losses
�	variables
�	keras_api
n
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�trainable_variables
�regularization_losses
�	variables
�	keras_api
V
�trainable_variables
�regularization_losses
�	variables
�	keras_api
n
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�
70
81
=2
>3
H4
I5
P6
Q7
[8
\9
g10
h11
m12
n13
x14
y15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
 
�
70
81
=2
>3
H4
I5
J6
K7
P8
Q9
[10
\11
]12
^13
g14
h15
m16
n17
x18
y19
z20
{21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53
�
�non_trainable_variables
�layers
&trainable_variables
 �layer_regularization_losses
�metrics
'regularization_losses
(	variables
 
 
 
 
�
�non_trainable_variables
�layers
+trainable_variables
 �layer_regularization_losses
�metrics
,regularization_losses
-	variables
 
 
 
�
�non_trainable_variables
�layers
/trainable_variables
 �layer_regularization_losses
�metrics
0regularization_losses
1	variables
 
 
 
�
�non_trainable_variables
�layers
3trainable_variables
 �layer_regularization_losses
�metrics
4regularization_losses
5	variables
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

70
81
 

70
81
�
�non_trainable_variables
�layers
9trainable_variables
 �layer_regularization_losses
�metrics
:regularization_losses
;	variables
db
VARIABLE_VALUEres_conv_A_repr_a/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEres_conv_A_repr_a/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

=0
>1
 

=0
>1
�
�non_trainable_variables
�layers
?trainable_variables
 �layer_regularization_losses
�metrics
@regularization_losses
A	variables
 
 
 
�
�non_trainable_variables
�layers
Ctrainable_variables
 �layer_regularization_losses
�metrics
Dregularization_losses
E	variables
 
db
VARIABLE_VALUEbatch_normalization/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEbatch_normalization/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE
pn
VARIABLE_VALUEbatch_normalization/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE#batch_normalization/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

H0
I1
 

H0
I1
J2
K3
�
�non_trainable_variables
�layers
Ltrainable_variables
 �layer_regularization_losses
�metrics
Mregularization_losses
N	variables
db
VARIABLE_VALUEres_conv_B_repr_a/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEres_conv_B_repr_a/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

P0
Q1
 

P0
Q1
�
�non_trainable_variables
�layers
Rtrainable_variables
 �layer_regularization_losses
�metrics
Sregularization_losses
T	variables
 
 
 
�
�non_trainable_variables
�layers
Vtrainable_variables
 �layer_regularization_losses
�metrics
Wregularization_losses
X	variables
 
fd
VARIABLE_VALUEbatch_normalization_1/gamma5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_1/beta4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_1/moving_mean;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_1/moving_variance?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

[0
\1
 

[0
\1
]2
^3
�
�non_trainable_variables
�layers
_trainable_variables
 �layer_regularization_losses
�metrics
`regularization_losses
a	variables
 
 
 
�
�non_trainable_variables
�layers
ctrainable_variables
 �layer_regularization_losses
�metrics
dregularization_losses
e	variables
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE

g0
h1
 

g0
h1
�
�non_trainable_variables
�layers
itrainable_variables
 �layer_regularization_losses
�metrics
jregularization_losses
k	variables
db
VARIABLE_VALUEres_conv_A_repr_b/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEres_conv_A_repr_b/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE

m0
n1
 

m0
n1
�
�non_trainable_variables
�layers
otrainable_variables
 �layer_regularization_losses
�metrics
pregularization_losses
q	variables
 
 
 
�
�non_trainable_variables
�layers
strainable_variables
 �layer_regularization_losses
�metrics
tregularization_losses
u	variables
 
fd
VARIABLE_VALUEbatch_normalization_2/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_2/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_2/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_2/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

x0
y1
 

x0
y1
z2
{3
�
�non_trainable_variables
�layers
|trainable_variables
 �layer_regularization_losses
�metrics
}regularization_losses
~	variables
db
VARIABLE_VALUEres_conv_B_repr_b/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
`^
VARIABLE_VALUEres_conv_B_repr_b/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 

�0
�1
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
 
 
 
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
 
fd
VARIABLE_VALUEbatch_normalization_3/gamma5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUE
db
VARIABLE_VALUEbatch_normalization_3/beta4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE!batch_normalization_3/moving_mean;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
zx
VARIABLE_VALUE%batch_normalization_3/moving_variance?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 
 
�0
�1
�2
�3
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
 
 
 
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
 
 
 
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
ec
VARIABLE_VALUEres_conv_A_repr_c/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEres_conv_A_repr_c/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 

�0
�1
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
 
 
 
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
 
ge
VARIABLE_VALUEbatch_normalization_4/gamma6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_4/beta5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_4/moving_mean<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_4/moving_variance@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 
 
�0
�1
�2
�3
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
ec
VARIABLE_VALUEres_conv_B_repr_c/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEres_conv_B_repr_c/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 

�0
�1
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
 
 
 
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
 
ge
VARIABLE_VALUEbatch_normalization_5/gamma6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_5/beta5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_5/moving_mean<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_5/moving_variance@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 
 
�0
�1
�2
�3
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
 
 
 
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
 
 
 
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
ec
VARIABLE_VALUEres_conv_A_repr_d/kernel7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEres_conv_A_repr_d/bias5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 

�0
�1
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
 
 
 
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
 
ge
VARIABLE_VALUEbatch_normalization_6/gamma6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_6/beta5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_6/moving_mean<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_6/moving_variance@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 
 
�0
�1
�2
�3
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
ec
VARIABLE_VALUEres_conv_B_repr_d/kernel7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEres_conv_B_repr_d/bias5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 

�0
�1
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
 
 
 
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
 
ge
VARIABLE_VALUEbatch_normalization_7/gamma6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEbatch_normalization_7/beta5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE!batch_normalization_7/moving_mean<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUE%batch_normalization_7/moving_variance@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 
 
�0
�1
�2
�3
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
 
 
 
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
^\
VARIABLE_VALUErepr_board/kernel7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUErepr_board/bias5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE

�0
�1
 

�0
�1
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
�
J0
K1
]2
^3
z4
{5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

J0
K1
 
 
 
 
 
 
 
 
 
 
 

]0
^1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

z0
{1
 
 
 
 
 
 
 
 
 
 
 

�0
�1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

�0
�1
 
 
 
 
 
 
 
 
 
 
 

�0
�1
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

�0
�1
 
 
 
 
 
 
 
 
 
 
 

�0
�1
 
 
 
 
 
 
 
 
 
 
 
�
serving_default_boardPlaceholder*3
_output_shapes!
:���������``*
dtype0*(
shape:���������``
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_boardconv2d/kernelconv2d/biasres_conv_A_repr_a/kernelres_conv_A_repr_a/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceres_conv_B_repr_a/kernelres_conv_B_repr_a/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_1/kernelconv2d_1/biasres_conv_A_repr_b/kernelres_conv_A_repr_b/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceres_conv_B_repr_b/kernelres_conv_B_repr_b/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceres_conv_A_repr_c/kernelres_conv_A_repr_c/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceres_conv_B_repr_c/kernelres_conv_B_repr_c/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceres_conv_A_repr_d/kernelres_conv_A_repr_d/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceres_conv_B_repr_d/kernelres_conv_B_repr_d/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variancerepr_board/kernelrepr_board/bias*B
Tin;
927*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������*/
config_proto

GPU

CPU2 *0J 8*,
f'R%
#__inference_signature_wrapper_25063
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp,res_conv_A_repr_a/kernel/Read/ReadVariableOp*res_conv_A_repr_a/bias/Read/ReadVariableOp-batch_normalization/gamma/Read/ReadVariableOp,batch_normalization/beta/Read/ReadVariableOp3batch_normalization/moving_mean/Read/ReadVariableOp7batch_normalization/moving_variance/Read/ReadVariableOp,res_conv_B_repr_a/kernel/Read/ReadVariableOp*res_conv_B_repr_a/bias/Read/ReadVariableOp/batch_normalization_1/gamma/Read/ReadVariableOp.batch_normalization_1/beta/Read/ReadVariableOp5batch_normalization_1/moving_mean/Read/ReadVariableOp9batch_normalization_1/moving_variance/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp,res_conv_A_repr_b/kernel/Read/ReadVariableOp*res_conv_A_repr_b/bias/Read/ReadVariableOp/batch_normalization_2/gamma/Read/ReadVariableOp.batch_normalization_2/beta/Read/ReadVariableOp5batch_normalization_2/moving_mean/Read/ReadVariableOp9batch_normalization_2/moving_variance/Read/ReadVariableOp,res_conv_B_repr_b/kernel/Read/ReadVariableOp*res_conv_B_repr_b/bias/Read/ReadVariableOp/batch_normalization_3/gamma/Read/ReadVariableOp.batch_normalization_3/beta/Read/ReadVariableOp5batch_normalization_3/moving_mean/Read/ReadVariableOp9batch_normalization_3/moving_variance/Read/ReadVariableOp,res_conv_A_repr_c/kernel/Read/ReadVariableOp*res_conv_A_repr_c/bias/Read/ReadVariableOp/batch_normalization_4/gamma/Read/ReadVariableOp.batch_normalization_4/beta/Read/ReadVariableOp5batch_normalization_4/moving_mean/Read/ReadVariableOp9batch_normalization_4/moving_variance/Read/ReadVariableOp,res_conv_B_repr_c/kernel/Read/ReadVariableOp*res_conv_B_repr_c/bias/Read/ReadVariableOp/batch_normalization_5/gamma/Read/ReadVariableOp.batch_normalization_5/beta/Read/ReadVariableOp5batch_normalization_5/moving_mean/Read/ReadVariableOp9batch_normalization_5/moving_variance/Read/ReadVariableOp,res_conv_A_repr_d/kernel/Read/ReadVariableOp*res_conv_A_repr_d/bias/Read/ReadVariableOp/batch_normalization_6/gamma/Read/ReadVariableOp.batch_normalization_6/beta/Read/ReadVariableOp5batch_normalization_6/moving_mean/Read/ReadVariableOp9batch_normalization_6/moving_variance/Read/ReadVariableOp,res_conv_B_repr_d/kernel/Read/ReadVariableOp*res_conv_B_repr_d/bias/Read/ReadVariableOp/batch_normalization_7/gamma/Read/ReadVariableOp.batch_normalization_7/beta/Read/ReadVariableOp5batch_normalization_7/moving_mean/Read/ReadVariableOp9batch_normalization_7/moving_variance/Read/ReadVariableOp%repr_board/kernel/Read/ReadVariableOp#repr_board/bias/Read/ReadVariableOpConst*C
Tin<
:28*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: */
config_proto

GPU

CPU2 *0J 8*'
f"R 
__inference__traced_save_27784
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasres_conv_A_repr_a/kernelres_conv_A_repr_a/biasbatch_normalization/gammabatch_normalization/betabatch_normalization/moving_mean#batch_normalization/moving_varianceres_conv_B_repr_a/kernelres_conv_B_repr_a/biasbatch_normalization_1/gammabatch_normalization_1/beta!batch_normalization_1/moving_mean%batch_normalization_1/moving_varianceconv2d_1/kernelconv2d_1/biasres_conv_A_repr_b/kernelres_conv_A_repr_b/biasbatch_normalization_2/gammabatch_normalization_2/beta!batch_normalization_2/moving_mean%batch_normalization_2/moving_varianceres_conv_B_repr_b/kernelres_conv_B_repr_b/biasbatch_normalization_3/gammabatch_normalization_3/beta!batch_normalization_3/moving_mean%batch_normalization_3/moving_varianceres_conv_A_repr_c/kernelres_conv_A_repr_c/biasbatch_normalization_4/gammabatch_normalization_4/beta!batch_normalization_4/moving_mean%batch_normalization_4/moving_varianceres_conv_B_repr_c/kernelres_conv_B_repr_c/biasbatch_normalization_5/gammabatch_normalization_5/beta!batch_normalization_5/moving_mean%batch_normalization_5/moving_varianceres_conv_A_repr_d/kernelres_conv_A_repr_d/biasbatch_normalization_6/gammabatch_normalization_6/beta!batch_normalization_6/moving_mean%batch_normalization_6/moving_varianceres_conv_B_repr_d/kernelres_conv_B_repr_d/biasbatch_normalization_7/gammabatch_normalization_7/beta!batch_normalization_7/moving_mean%batch_normalization_7/moving_variancerepr_board/kernelrepr_board/bias*B
Tin;
927*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: */
config_proto

GPU

CPU2 *0J 8**
f%R#
!__inference__traced_restore_27958��0
�
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_26244

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������00 : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������00 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������00 ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�

�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_22195

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
�%
�
.__inference_Representation_layer_call_fn_25941

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36#
statefulpartitionedcall_args_37#
statefulpartitionedcall_args_38#
statefulpartitionedcall_args_39#
statefulpartitionedcall_args_40#
statefulpartitionedcall_args_41#
statefulpartitionedcall_args_42#
statefulpartitionedcall_args_43#
statefulpartitionedcall_args_44#
statefulpartitionedcall_args_45#
statefulpartitionedcall_args_46#
statefulpartitionedcall_args_47#
statefulpartitionedcall_args_48#
statefulpartitionedcall_args_49#
statefulpartitionedcall_args_50#
statefulpartitionedcall_args_51#
statefulpartitionedcall_args_52#
statefulpartitionedcall_args_53#
statefulpartitionedcall_args_54
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36statefulpartitionedcall_args_37statefulpartitionedcall_args_38statefulpartitionedcall_args_39statefulpartitionedcall_args_40statefulpartitionedcall_args_41statefulpartitionedcall_args_42statefulpartitionedcall_args_43statefulpartitionedcall_args_44statefulpartitionedcall_args_45statefulpartitionedcall_args_46statefulpartitionedcall_args_47statefulpartitionedcall_args_48statefulpartitionedcall_args_49statefulpartitionedcall_args_50statefulpartitionedcall_args_51statefulpartitionedcall_args_52statefulpartitionedcall_args_53statefulpartitionedcall_args_54*B
Tin;
927*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������*/
config_proto

GPU

CPU2 *0J 8*R
fMRK
I__inference_Representation_layer_call_and_return_conditional_losses_247702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������``::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
__inference_loss_fn_8_27585@
<repr_board_kernel_regularizer_square_readvariableop_resource
identity��3repr_board/kernel/Regularizer/Square/ReadVariableOp�
3repr_board/kernel/Regularizer/Square/ReadVariableOpReadVariableOp<repr_board_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@*
dtype025
3repr_board/kernel/Regularizer/Square/ReadVariableOp�
$repr_board/kernel/Regularizer/SquareSquare;repr_board/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2&
$repr_board/kernel/Regularizer/Square�
#repr_board/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2%
#repr_board/kernel/Regularizer/Const�
!repr_board/kernel/Regularizer/SumSum(repr_board/kernel/Regularizer/Square:y:0,repr_board/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!repr_board/kernel/Regularizer/Sum�
#repr_board/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72%
#repr_board/kernel/Regularizer/mul/x�
!repr_board/kernel/Regularizer/mulMul,repr_board/kernel/Regularizer/mul/x:output:0*repr_board/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!repr_board/kernel/Regularizer/mul�
#repr_board/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#repr_board/kernel/Regularizer/add/x�
!repr_board/kernel/Regularizer/addAddV2,repr_board/kernel/Regularizer/add/x:output:0%repr_board/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2#
!repr_board/kernel/Regularizer/add�
IdentityIdentity%repr_board/kernel/Regularizer/add:z:04^repr_board/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2j
3repr_board/kernel/Regularizer/Square/ReadVariableOp3repr_board/kernel/Regularizer/Square/ReadVariableOp
�%
�
.__inference_Representation_layer_call_fn_25882

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36#
statefulpartitionedcall_args_37#
statefulpartitionedcall_args_38#
statefulpartitionedcall_args_39#
statefulpartitionedcall_args_40#
statefulpartitionedcall_args_41#
statefulpartitionedcall_args_42#
statefulpartitionedcall_args_43#
statefulpartitionedcall_args_44#
statefulpartitionedcall_args_45#
statefulpartitionedcall_args_46#
statefulpartitionedcall_args_47#
statefulpartitionedcall_args_48#
statefulpartitionedcall_args_49#
statefulpartitionedcall_args_50#
statefulpartitionedcall_args_51#
statefulpartitionedcall_args_52#
statefulpartitionedcall_args_53#
statefulpartitionedcall_args_54
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36statefulpartitionedcall_args_37statefulpartitionedcall_args_38statefulpartitionedcall_args_39statefulpartitionedcall_args_40statefulpartitionedcall_args_41statefulpartitionedcall_args_42statefulpartitionedcall_args_43statefulpartitionedcall_args_44statefulpartitionedcall_args_45statefulpartitionedcall_args_46statefulpartitionedcall_args_47statefulpartitionedcall_args_48statefulpartitionedcall_args_49statefulpartitionedcall_args_50statefulpartitionedcall_args_51statefulpartitionedcall_args_52statefulpartitionedcall_args_53statefulpartitionedcall_args_54*B
Tin;
927*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������*/
config_proto

GPU

CPU2 *0J 8*R
fMRK
I__inference_Representation_layer_call_and_return_conditional_losses_245372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������``::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
c
G__inference_activation_6_layer_call_and_return_conditional_losses_23905

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�#
�
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_23946

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_23931
assignmovingavg_1_23938
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/23931*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/23931*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23931*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/23931*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/23931*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23931AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/23931*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/23938*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/23938*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23938*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/23938*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/23938*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23938AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/23938*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_23348

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������00 : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������00 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������00 ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
j
>__inference_add_layer_call_and_return_conditional_losses_26342
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������00 2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������00 2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������00 :���������00 :( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
��
�#
I__inference_Representation_layer_call_and_return_conditional_losses_24770

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_24
0res_conv_a_repr_a_statefulpartitionedcall_args_14
0res_conv_a_repr_a_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_16
2batch_normalization_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_36
2batch_normalization_statefulpartitionedcall_args_44
0res_conv_b_repr_a_statefulpartitionedcall_args_14
0res_conv_b_repr_a_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_18
4batch_normalization_1_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_38
4batch_normalization_1_statefulpartitionedcall_args_4+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_24
0res_conv_a_repr_b_statefulpartitionedcall_args_14
0res_conv_a_repr_b_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_18
4batch_normalization_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_38
4batch_normalization_2_statefulpartitionedcall_args_44
0res_conv_b_repr_b_statefulpartitionedcall_args_14
0res_conv_b_repr_b_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_18
4batch_normalization_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_38
4batch_normalization_3_statefulpartitionedcall_args_44
0res_conv_a_repr_c_statefulpartitionedcall_args_14
0res_conv_a_repr_c_statefulpartitionedcall_args_28
4batch_normalization_4_statefulpartitionedcall_args_18
4batch_normalization_4_statefulpartitionedcall_args_28
4batch_normalization_4_statefulpartitionedcall_args_38
4batch_normalization_4_statefulpartitionedcall_args_44
0res_conv_b_repr_c_statefulpartitionedcall_args_14
0res_conv_b_repr_c_statefulpartitionedcall_args_28
4batch_normalization_5_statefulpartitionedcall_args_18
4batch_normalization_5_statefulpartitionedcall_args_28
4batch_normalization_5_statefulpartitionedcall_args_38
4batch_normalization_5_statefulpartitionedcall_args_44
0res_conv_a_repr_d_statefulpartitionedcall_args_14
0res_conv_a_repr_d_statefulpartitionedcall_args_28
4batch_normalization_6_statefulpartitionedcall_args_18
4batch_normalization_6_statefulpartitionedcall_args_28
4batch_normalization_6_statefulpartitionedcall_args_38
4batch_normalization_6_statefulpartitionedcall_args_44
0res_conv_b_repr_d_statefulpartitionedcall_args_14
0res_conv_b_repr_d_statefulpartitionedcall_args_28
4batch_normalization_7_statefulpartitionedcall_args_18
4batch_normalization_7_statefulpartitionedcall_args_28
4batch_normalization_7_statefulpartitionedcall_args_38
4batch_normalization_7_statefulpartitionedcall_args_4-
)repr_board_statefulpartitionedcall_args_1-
)repr_board_statefulpartitionedcall_args_2
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�"repr_board/StatefulPartitionedCall�1repr_board/bias/Regularizer/Square/ReadVariableOp�3repr_board/kernel/Regularizer/Square/ReadVariableOp�)res_conv_A_repr_a/StatefulPartitionedCall�:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp�)res_conv_A_repr_b/StatefulPartitionedCall�:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp�)res_conv_A_repr_c/StatefulPartitionedCall�:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp�)res_conv_A_repr_d/StatefulPartitionedCall�:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp�)res_conv_B_repr_a/StatefulPartitionedCall�:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp�)res_conv_B_repr_b/StatefulPartitionedCall�:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp�)res_conv_B_repr_c/StatefulPartitionedCall�:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp�)res_conv_B_repr_d/StatefulPartitionedCall�:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp�
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:���������``*/
config_proto

GPU

CPU2 *0J 8*K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_232432
reshape/PartitionedCall�
permute/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:���������``*/
config_proto

GPU

CPU2 *0J 8*K
fFRD
B__inference_permute_layer_call_and_return_conditional_losses_218372
permute/PartitionedCall�
reshape_1/PartitionedCallPartitionedCall permute/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������``*/
config_proto

GPU

CPU2 *0J 8*M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_232662
reshape_1/PartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_218552 
conv2d/StatefulPartitionedCall�
)res_conv_A_repr_a/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:00res_conv_a_repr_a_statefulpartitionedcall_args_10res_conv_a_repr_a_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_A_repr_a_layer_call_and_return_conditional_losses_218832+
)res_conv_A_repr_a/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall2res_conv_A_repr_a/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_232852
activation/PartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:02batch_normalization_statefulpartitionedcall_args_12batch_normalization_statefulpartitionedcall_args_22batch_normalization_statefulpartitionedcall_args_32batch_normalization_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_233482-
+batch_normalization/StatefulPartitionedCall�
)res_conv_B_repr_a/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:00res_conv_b_repr_a_statefulpartitionedcall_args_10res_conv_b_repr_a_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_B_repr_a_layer_call_and_return_conditional_losses_220432+
)res_conv_B_repr_a/StatefulPartitionedCall�
activation_1/PartitionedCallPartitionedCall2res_conv_B_repr_a/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_233802
activation_1/PartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:04batch_normalization_1_statefulpartitionedcall_args_14batch_normalization_1_statefulpartitionedcall_args_24batch_normalization_1_statefulpartitionedcall_args_34batch_normalization_1_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_234432/
-batch_normalization_1/StatefulPartitionedCall�
add/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_234732
add/PartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_221952"
 conv2d_1/StatefulPartitionedCall�
)res_conv_A_repr_b/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:00res_conv_a_repr_b_statefulpartitionedcall_args_10res_conv_a_repr_b_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_A_repr_b_layer_call_and_return_conditional_losses_222232+
)res_conv_A_repr_b/StatefulPartitionedCall�
activation_2/PartitionedCallPartitionedCall2res_conv_A_repr_b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_234932
activation_2/PartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:04batch_normalization_2_statefulpartitionedcall_args_14batch_normalization_2_statefulpartitionedcall_args_24batch_normalization_2_statefulpartitionedcall_args_34batch_normalization_2_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_235562/
-batch_normalization_2/StatefulPartitionedCall�
)res_conv_B_repr_b/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:00res_conv_b_repr_b_statefulpartitionedcall_args_10res_conv_b_repr_b_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_B_repr_b_layer_call_and_return_conditional_losses_223832+
)res_conv_B_repr_b/StatefulPartitionedCall�
activation_3/PartitionedCallPartitionedCall2res_conv_B_repr_b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_235882
activation_3/PartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:04batch_normalization_3_statefulpartitionedcall_args_14batch_normalization_3_statefulpartitionedcall_args_24batch_normalization_3_statefulpartitionedcall_args_34batch_normalization_3_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_236512/
-batch_normalization_3/StatefulPartitionedCall�
add_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_236812
add_1/PartitionedCall�
!average_pooling2d/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_225292#
!average_pooling2d/PartitionedCall�
)res_conv_A_repr_c/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:00res_conv_a_repr_c_statefulpartitionedcall_args_10res_conv_a_repr_c_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_A_repr_c_layer_call_and_return_conditional_losses_225552+
)res_conv_A_repr_c/StatefulPartitionedCall�
activation_4/PartitionedCallPartitionedCall2res_conv_A_repr_c/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_236992
activation_4/PartitionedCall�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:04batch_normalization_4_statefulpartitionedcall_args_14batch_normalization_4_statefulpartitionedcall_args_24batch_normalization_4_statefulpartitionedcall_args_34batch_normalization_4_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_237622/
-batch_normalization_4/StatefulPartitionedCall�
)res_conv_B_repr_c/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:00res_conv_b_repr_c_statefulpartitionedcall_args_10res_conv_b_repr_c_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_B_repr_c_layer_call_and_return_conditional_losses_227152+
)res_conv_B_repr_c/StatefulPartitionedCall�
activation_5/PartitionedCallPartitionedCall2res_conv_B_repr_c/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_237942
activation_5/PartitionedCall�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:04batch_normalization_5_statefulpartitionedcall_args_14batch_normalization_5_statefulpartitionedcall_args_24batch_normalization_5_statefulpartitionedcall_args_34batch_normalization_5_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_238572/
-batch_normalization_5/StatefulPartitionedCall�
add_2/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*I
fDRB
@__inference_add_2_layer_call_and_return_conditional_losses_238872
add_2/PartitionedCall�
#average_pooling2d_1/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*W
fRRP
N__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_228612%
#average_pooling2d_1/PartitionedCall�
)res_conv_A_repr_d/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_1/PartitionedCall:output:00res_conv_a_repr_d_statefulpartitionedcall_args_10res_conv_a_repr_d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_A_repr_d_layer_call_and_return_conditional_losses_228872+
)res_conv_A_repr_d/StatefulPartitionedCall�
activation_6/PartitionedCallPartitionedCall2res_conv_A_repr_d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_6_layer_call_and_return_conditional_losses_239052
activation_6/PartitionedCall�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:04batch_normalization_6_statefulpartitionedcall_args_14batch_normalization_6_statefulpartitionedcall_args_24batch_normalization_6_statefulpartitionedcall_args_34batch_normalization_6_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_239682/
-batch_normalization_6/StatefulPartitionedCall�
)res_conv_B_repr_d/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:00res_conv_b_repr_d_statefulpartitionedcall_args_10res_conv_b_repr_d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_B_repr_d_layer_call_and_return_conditional_losses_230472+
)res_conv_B_repr_d/StatefulPartitionedCall�
activation_7/PartitionedCallPartitionedCall2res_conv_B_repr_d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_7_layer_call_and_return_conditional_losses_240002
activation_7/PartitionedCall�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:04batch_normalization_7_statefulpartitionedcall_args_14batch_normalization_7_statefulpartitionedcall_args_24batch_normalization_7_statefulpartitionedcall_args_34batch_normalization_7_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_240632/
-batch_normalization_7/StatefulPartitionedCall�
add_3/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0,average_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*I
fDRB
@__inference_add_3_layer_call_and_return_conditional_losses_240932
add_3/PartitionedCall�
"repr_board/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0)repr_board_statefulpartitionedcall_args_1)repr_board_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������*/
config_proto

GPU

CPU2 *0J 8*N
fIRG
E__inference_repr_board_layer_call_and_return_conditional_losses_232162$
"repr_board/StatefulPartitionedCall�
:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_a_repr_a_statefulpartitionedcall_args_1*^res_conv_A_repr_a/StatefulPartitionedCall*&
_output_shapes
:  *
dtype02<
:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_a/kernel/Regularizer/SquareSquareBres_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2-
+res_conv_A_repr_a/kernel/Regularizer/Square�
*res_conv_A_repr_a/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_a/kernel/Regularizer/Const�
(res_conv_A_repr_a/kernel/Regularizer/SumSum/res_conv_A_repr_a/kernel/Regularizer/Square:y:03res_conv_A_repr_a/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_a/kernel/Regularizer/Sum�
*res_conv_A_repr_a/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_a/kernel/Regularizer/mul/x�
(res_conv_A_repr_a/kernel/Regularizer/mulMul3res_conv_A_repr_a/kernel/Regularizer/mul/x:output:01res_conv_A_repr_a/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_a/kernel/Regularizer/mul�
*res_conv_A_repr_a/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_a/kernel/Regularizer/add/x�
(res_conv_A_repr_a/kernel/Regularizer/addAddV23res_conv_A_repr_a/kernel/Regularizer/add/x:output:0,res_conv_A_repr_a/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_a/kernel/Regularizer/add�
:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_b_repr_a_statefulpartitionedcall_args_1*^res_conv_B_repr_a/StatefulPartitionedCall*&
_output_shapes
:  *
dtype02<
:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_a/kernel/Regularizer/SquareSquareBres_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2-
+res_conv_B_repr_a/kernel/Regularizer/Square�
*res_conv_B_repr_a/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_a/kernel/Regularizer/Const�
(res_conv_B_repr_a/kernel/Regularizer/SumSum/res_conv_B_repr_a/kernel/Regularizer/Square:y:03res_conv_B_repr_a/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_a/kernel/Regularizer/Sum�
*res_conv_B_repr_a/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_a/kernel/Regularizer/mul/x�
(res_conv_B_repr_a/kernel/Regularizer/mulMul3res_conv_B_repr_a/kernel/Regularizer/mul/x:output:01res_conv_B_repr_a/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_a/kernel/Regularizer/mul�
*res_conv_B_repr_a/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_a/kernel/Regularizer/add/x�
(res_conv_B_repr_a/kernel/Regularizer/addAddV23res_conv_B_repr_a/kernel/Regularizer/add/x:output:0,res_conv_B_repr_a/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_a/kernel/Regularizer/add�
:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_a_repr_b_statefulpartitionedcall_args_1*^res_conv_A_repr_b/StatefulPartitionedCall*&
_output_shapes
:@@*
dtype02<
:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_b/kernel/Regularizer/SquareSquareBres_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_A_repr_b/kernel/Regularizer/Square�
*res_conv_A_repr_b/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_b/kernel/Regularizer/Const�
(res_conv_A_repr_b/kernel/Regularizer/SumSum/res_conv_A_repr_b/kernel/Regularizer/Square:y:03res_conv_A_repr_b/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_b/kernel/Regularizer/Sum�
*res_conv_A_repr_b/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_b/kernel/Regularizer/mul/x�
(res_conv_A_repr_b/kernel/Regularizer/mulMul3res_conv_A_repr_b/kernel/Regularizer/mul/x:output:01res_conv_A_repr_b/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_b/kernel/Regularizer/mul�
*res_conv_A_repr_b/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_b/kernel/Regularizer/add/x�
(res_conv_A_repr_b/kernel/Regularizer/addAddV23res_conv_A_repr_b/kernel/Regularizer/add/x:output:0,res_conv_A_repr_b/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_b/kernel/Regularizer/add�
:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_b_repr_b_statefulpartitionedcall_args_1*^res_conv_B_repr_b/StatefulPartitionedCall*&
_output_shapes
:@@*
dtype02<
:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_b/kernel/Regularizer/SquareSquareBres_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_B_repr_b/kernel/Regularizer/Square�
*res_conv_B_repr_b/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_b/kernel/Regularizer/Const�
(res_conv_B_repr_b/kernel/Regularizer/SumSum/res_conv_B_repr_b/kernel/Regularizer/Square:y:03res_conv_B_repr_b/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_b/kernel/Regularizer/Sum�
*res_conv_B_repr_b/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_b/kernel/Regularizer/mul/x�
(res_conv_B_repr_b/kernel/Regularizer/mulMul3res_conv_B_repr_b/kernel/Regularizer/mul/x:output:01res_conv_B_repr_b/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_b/kernel/Regularizer/mul�
*res_conv_B_repr_b/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_b/kernel/Regularizer/add/x�
(res_conv_B_repr_b/kernel/Regularizer/addAddV23res_conv_B_repr_b/kernel/Regularizer/add/x:output:0,res_conv_B_repr_b/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_b/kernel/Regularizer/add�
:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_a_repr_c_statefulpartitionedcall_args_1*^res_conv_A_repr_c/StatefulPartitionedCall*&
_output_shapes
:@@*
dtype02<
:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_c/kernel/Regularizer/SquareSquareBres_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_A_repr_c/kernel/Regularizer/Square�
*res_conv_A_repr_c/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_c/kernel/Regularizer/Const�
(res_conv_A_repr_c/kernel/Regularizer/SumSum/res_conv_A_repr_c/kernel/Regularizer/Square:y:03res_conv_A_repr_c/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_c/kernel/Regularizer/Sum�
*res_conv_A_repr_c/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_c/kernel/Regularizer/mul/x�
(res_conv_A_repr_c/kernel/Regularizer/mulMul3res_conv_A_repr_c/kernel/Regularizer/mul/x:output:01res_conv_A_repr_c/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_c/kernel/Regularizer/mul�
*res_conv_A_repr_c/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_c/kernel/Regularizer/add/x�
(res_conv_A_repr_c/kernel/Regularizer/addAddV23res_conv_A_repr_c/kernel/Regularizer/add/x:output:0,res_conv_A_repr_c/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_c/kernel/Regularizer/add�
:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_b_repr_c_statefulpartitionedcall_args_1*^res_conv_B_repr_c/StatefulPartitionedCall*&
_output_shapes
:@@*
dtype02<
:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_c/kernel/Regularizer/SquareSquareBres_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_B_repr_c/kernel/Regularizer/Square�
*res_conv_B_repr_c/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_c/kernel/Regularizer/Const�
(res_conv_B_repr_c/kernel/Regularizer/SumSum/res_conv_B_repr_c/kernel/Regularizer/Square:y:03res_conv_B_repr_c/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_c/kernel/Regularizer/Sum�
*res_conv_B_repr_c/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_c/kernel/Regularizer/mul/x�
(res_conv_B_repr_c/kernel/Regularizer/mulMul3res_conv_B_repr_c/kernel/Regularizer/mul/x:output:01res_conv_B_repr_c/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_c/kernel/Regularizer/mul�
*res_conv_B_repr_c/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_c/kernel/Regularizer/add/x�
(res_conv_B_repr_c/kernel/Regularizer/addAddV23res_conv_B_repr_c/kernel/Regularizer/add/x:output:0,res_conv_B_repr_c/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_c/kernel/Regularizer/add�
:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_a_repr_d_statefulpartitionedcall_args_1*^res_conv_A_repr_d/StatefulPartitionedCall*&
_output_shapes
:@@*
dtype02<
:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_d/kernel/Regularizer/SquareSquareBres_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_A_repr_d/kernel/Regularizer/Square�
*res_conv_A_repr_d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_d/kernel/Regularizer/Const�
(res_conv_A_repr_d/kernel/Regularizer/SumSum/res_conv_A_repr_d/kernel/Regularizer/Square:y:03res_conv_A_repr_d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_d/kernel/Regularizer/Sum�
*res_conv_A_repr_d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_d/kernel/Regularizer/mul/x�
(res_conv_A_repr_d/kernel/Regularizer/mulMul3res_conv_A_repr_d/kernel/Regularizer/mul/x:output:01res_conv_A_repr_d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_d/kernel/Regularizer/mul�
*res_conv_A_repr_d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_d/kernel/Regularizer/add/x�
(res_conv_A_repr_d/kernel/Regularizer/addAddV23res_conv_A_repr_d/kernel/Regularizer/add/x:output:0,res_conv_A_repr_d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_d/kernel/Regularizer/add�
:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_b_repr_d_statefulpartitionedcall_args_1*^res_conv_B_repr_d/StatefulPartitionedCall*&
_output_shapes
:@@*
dtype02<
:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_d/kernel/Regularizer/SquareSquareBres_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_B_repr_d/kernel/Regularizer/Square�
*res_conv_B_repr_d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_d/kernel/Regularizer/Const�
(res_conv_B_repr_d/kernel/Regularizer/SumSum/res_conv_B_repr_d/kernel/Regularizer/Square:y:03res_conv_B_repr_d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_d/kernel/Regularizer/Sum�
*res_conv_B_repr_d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_d/kernel/Regularizer/mul/x�
(res_conv_B_repr_d/kernel/Regularizer/mulMul3res_conv_B_repr_d/kernel/Regularizer/mul/x:output:01res_conv_B_repr_d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_d/kernel/Regularizer/mul�
*res_conv_B_repr_d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_d/kernel/Regularizer/add/x�
(res_conv_B_repr_d/kernel/Regularizer/addAddV23res_conv_B_repr_d/kernel/Regularizer/add/x:output:0,res_conv_B_repr_d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_d/kernel/Regularizer/add�
3repr_board/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)repr_board_statefulpartitionedcall_args_1#^repr_board/StatefulPartitionedCall*&
_output_shapes
:@*
dtype025
3repr_board/kernel/Regularizer/Square/ReadVariableOp�
$repr_board/kernel/Regularizer/SquareSquare;repr_board/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2&
$repr_board/kernel/Regularizer/Square�
#repr_board/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2%
#repr_board/kernel/Regularizer/Const�
!repr_board/kernel/Regularizer/SumSum(repr_board/kernel/Regularizer/Square:y:0,repr_board/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!repr_board/kernel/Regularizer/Sum�
#repr_board/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72%
#repr_board/kernel/Regularizer/mul/x�
!repr_board/kernel/Regularizer/mulMul,repr_board/kernel/Regularizer/mul/x:output:0*repr_board/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!repr_board/kernel/Regularizer/mul�
#repr_board/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#repr_board/kernel/Regularizer/add/x�
!repr_board/kernel/Regularizer/addAddV2,repr_board/kernel/Regularizer/add/x:output:0%repr_board/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2#
!repr_board/kernel/Regularizer/add�
1repr_board/bias/Regularizer/Square/ReadVariableOpReadVariableOp)repr_board_statefulpartitionedcall_args_2#^repr_board/StatefulPartitionedCall*
_output_shapes
:*
dtype023
1repr_board/bias/Regularizer/Square/ReadVariableOp�
"repr_board/bias/Regularizer/SquareSquare9repr_board/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2$
"repr_board/bias/Regularizer/Square�
!repr_board/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!repr_board/bias/Regularizer/Const�
repr_board/bias/Regularizer/SumSum&repr_board/bias/Regularizer/Square:y:0*repr_board/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
repr_board/bias/Regularizer/Sum�
!repr_board/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72#
!repr_board/bias/Regularizer/mul/x�
repr_board/bias/Regularizer/mulMul*repr_board/bias/Regularizer/mul/x:output:0(repr_board/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
repr_board/bias/Regularizer/mul�
!repr_board/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!repr_board/bias/Regularizer/add/x�
repr_board/bias/Regularizer/addAddV2*repr_board/bias/Regularizer/add/x:output:0#repr_board/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
repr_board/bias/Regularizer/add�
IdentityIdentity+repr_board/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall#^repr_board/StatefulPartitionedCall2^repr_board/bias/Regularizer/Square/ReadVariableOp4^repr_board/kernel/Regularizer/Square/ReadVariableOp*^res_conv_A_repr_a/StatefulPartitionedCall;^res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp*^res_conv_A_repr_b/StatefulPartitionedCall;^res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp*^res_conv_A_repr_c/StatefulPartitionedCall;^res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp*^res_conv_A_repr_d/StatefulPartitionedCall;^res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp*^res_conv_B_repr_a/StatefulPartitionedCall;^res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp*^res_conv_B_repr_b/StatefulPartitionedCall;^res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp*^res_conv_B_repr_c/StatefulPartitionedCall;^res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp*^res_conv_B_repr_d/StatefulPartitionedCall;^res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������``::::::::::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2H
"repr_board/StatefulPartitionedCall"repr_board/StatefulPartitionedCall2f
1repr_board/bias/Regularizer/Square/ReadVariableOp1repr_board/bias/Regularizer/Square/ReadVariableOp2j
3repr_board/kernel/Regularizer/Square/ReadVariableOp3repr_board/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_A_repr_a/StatefulPartitionedCall)res_conv_A_repr_a/StatefulPartitionedCall2x
:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_A_repr_b/StatefulPartitionedCall)res_conv_A_repr_b/StatefulPartitionedCall2x
:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_A_repr_c/StatefulPartitionedCall)res_conv_A_repr_c/StatefulPartitionedCall2x
:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_A_repr_d/StatefulPartitionedCall)res_conv_A_repr_d/StatefulPartitionedCall2x
:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_B_repr_a/StatefulPartitionedCall)res_conv_B_repr_a/StatefulPartitionedCall2x
:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_B_repr_b/StatefulPartitionedCall)res_conv_B_repr_b/StatefulPartitionedCall2x
:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_B_repr_c/StatefulPartitionedCall)res_conv_B_repr_c/StatefulPartitionedCall2x
:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_B_repr_d/StatefulPartitionedCall)res_conv_B_repr_d/StatefulPartitionedCall2x
:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_26140

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
j
@__inference_add_2_layer_call_and_return_conditional_losses_23887

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������@:���������@:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_27244

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
��
�#
I__inference_Representation_layer_call_and_return_conditional_losses_24537

inputs)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_24
0res_conv_a_repr_a_statefulpartitionedcall_args_14
0res_conv_a_repr_a_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_16
2batch_normalization_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_36
2batch_normalization_statefulpartitionedcall_args_44
0res_conv_b_repr_a_statefulpartitionedcall_args_14
0res_conv_b_repr_a_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_18
4batch_normalization_1_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_38
4batch_normalization_1_statefulpartitionedcall_args_4+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_24
0res_conv_a_repr_b_statefulpartitionedcall_args_14
0res_conv_a_repr_b_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_18
4batch_normalization_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_38
4batch_normalization_2_statefulpartitionedcall_args_44
0res_conv_b_repr_b_statefulpartitionedcall_args_14
0res_conv_b_repr_b_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_18
4batch_normalization_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_38
4batch_normalization_3_statefulpartitionedcall_args_44
0res_conv_a_repr_c_statefulpartitionedcall_args_14
0res_conv_a_repr_c_statefulpartitionedcall_args_28
4batch_normalization_4_statefulpartitionedcall_args_18
4batch_normalization_4_statefulpartitionedcall_args_28
4batch_normalization_4_statefulpartitionedcall_args_38
4batch_normalization_4_statefulpartitionedcall_args_44
0res_conv_b_repr_c_statefulpartitionedcall_args_14
0res_conv_b_repr_c_statefulpartitionedcall_args_28
4batch_normalization_5_statefulpartitionedcall_args_18
4batch_normalization_5_statefulpartitionedcall_args_28
4batch_normalization_5_statefulpartitionedcall_args_38
4batch_normalization_5_statefulpartitionedcall_args_44
0res_conv_a_repr_d_statefulpartitionedcall_args_14
0res_conv_a_repr_d_statefulpartitionedcall_args_28
4batch_normalization_6_statefulpartitionedcall_args_18
4batch_normalization_6_statefulpartitionedcall_args_28
4batch_normalization_6_statefulpartitionedcall_args_38
4batch_normalization_6_statefulpartitionedcall_args_44
0res_conv_b_repr_d_statefulpartitionedcall_args_14
0res_conv_b_repr_d_statefulpartitionedcall_args_28
4batch_normalization_7_statefulpartitionedcall_args_18
4batch_normalization_7_statefulpartitionedcall_args_28
4batch_normalization_7_statefulpartitionedcall_args_38
4batch_normalization_7_statefulpartitionedcall_args_4-
)repr_board_statefulpartitionedcall_args_1-
)repr_board_statefulpartitionedcall_args_2
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�"repr_board/StatefulPartitionedCall�1repr_board/bias/Regularizer/Square/ReadVariableOp�3repr_board/kernel/Regularizer/Square/ReadVariableOp�)res_conv_A_repr_a/StatefulPartitionedCall�:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp�)res_conv_A_repr_b/StatefulPartitionedCall�:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp�)res_conv_A_repr_c/StatefulPartitionedCall�:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp�)res_conv_A_repr_d/StatefulPartitionedCall�:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp�)res_conv_B_repr_a/StatefulPartitionedCall�:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp�)res_conv_B_repr_b/StatefulPartitionedCall�:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp�)res_conv_B_repr_c/StatefulPartitionedCall�:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp�)res_conv_B_repr_d/StatefulPartitionedCall�:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp�
reshape/PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:���������``*/
config_proto

GPU

CPU2 *0J 8*K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_232432
reshape/PartitionedCall�
permute/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:���������``*/
config_proto

GPU

CPU2 *0J 8*K
fFRD
B__inference_permute_layer_call_and_return_conditional_losses_218372
permute/PartitionedCall�
reshape_1/PartitionedCallPartitionedCall permute/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������``*/
config_proto

GPU

CPU2 *0J 8*M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_232662
reshape_1/PartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_218552 
conv2d/StatefulPartitionedCall�
)res_conv_A_repr_a/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:00res_conv_a_repr_a_statefulpartitionedcall_args_10res_conv_a_repr_a_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_A_repr_a_layer_call_and_return_conditional_losses_218832+
)res_conv_A_repr_a/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall2res_conv_A_repr_a/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_232852
activation/PartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:02batch_normalization_statefulpartitionedcall_args_12batch_normalization_statefulpartitionedcall_args_22batch_normalization_statefulpartitionedcall_args_32batch_normalization_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_233262-
+batch_normalization/StatefulPartitionedCall�
)res_conv_B_repr_a/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:00res_conv_b_repr_a_statefulpartitionedcall_args_10res_conv_b_repr_a_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_B_repr_a_layer_call_and_return_conditional_losses_220432+
)res_conv_B_repr_a/StatefulPartitionedCall�
activation_1/PartitionedCallPartitionedCall2res_conv_B_repr_a/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_233802
activation_1/PartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:04batch_normalization_1_statefulpartitionedcall_args_14batch_normalization_1_statefulpartitionedcall_args_24batch_normalization_1_statefulpartitionedcall_args_34batch_normalization_1_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_234212/
-batch_normalization_1/StatefulPartitionedCall�
add/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_234732
add/PartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_221952"
 conv2d_1/StatefulPartitionedCall�
)res_conv_A_repr_b/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:00res_conv_a_repr_b_statefulpartitionedcall_args_10res_conv_a_repr_b_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_A_repr_b_layer_call_and_return_conditional_losses_222232+
)res_conv_A_repr_b/StatefulPartitionedCall�
activation_2/PartitionedCallPartitionedCall2res_conv_A_repr_b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_234932
activation_2/PartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:04batch_normalization_2_statefulpartitionedcall_args_14batch_normalization_2_statefulpartitionedcall_args_24batch_normalization_2_statefulpartitionedcall_args_34batch_normalization_2_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_235342/
-batch_normalization_2/StatefulPartitionedCall�
)res_conv_B_repr_b/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:00res_conv_b_repr_b_statefulpartitionedcall_args_10res_conv_b_repr_b_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_B_repr_b_layer_call_and_return_conditional_losses_223832+
)res_conv_B_repr_b/StatefulPartitionedCall�
activation_3/PartitionedCallPartitionedCall2res_conv_B_repr_b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_235882
activation_3/PartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:04batch_normalization_3_statefulpartitionedcall_args_14batch_normalization_3_statefulpartitionedcall_args_24batch_normalization_3_statefulpartitionedcall_args_34batch_normalization_3_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_236292/
-batch_normalization_3/StatefulPartitionedCall�
add_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_236812
add_1/PartitionedCall�
!average_pooling2d/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_225292#
!average_pooling2d/PartitionedCall�
)res_conv_A_repr_c/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:00res_conv_a_repr_c_statefulpartitionedcall_args_10res_conv_a_repr_c_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_A_repr_c_layer_call_and_return_conditional_losses_225552+
)res_conv_A_repr_c/StatefulPartitionedCall�
activation_4/PartitionedCallPartitionedCall2res_conv_A_repr_c/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_236992
activation_4/PartitionedCall�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:04batch_normalization_4_statefulpartitionedcall_args_14batch_normalization_4_statefulpartitionedcall_args_24batch_normalization_4_statefulpartitionedcall_args_34batch_normalization_4_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_237402/
-batch_normalization_4/StatefulPartitionedCall�
)res_conv_B_repr_c/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:00res_conv_b_repr_c_statefulpartitionedcall_args_10res_conv_b_repr_c_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_B_repr_c_layer_call_and_return_conditional_losses_227152+
)res_conv_B_repr_c/StatefulPartitionedCall�
activation_5/PartitionedCallPartitionedCall2res_conv_B_repr_c/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_237942
activation_5/PartitionedCall�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:04batch_normalization_5_statefulpartitionedcall_args_14batch_normalization_5_statefulpartitionedcall_args_24batch_normalization_5_statefulpartitionedcall_args_34batch_normalization_5_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_238352/
-batch_normalization_5/StatefulPartitionedCall�
add_2/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*I
fDRB
@__inference_add_2_layer_call_and_return_conditional_losses_238872
add_2/PartitionedCall�
#average_pooling2d_1/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*W
fRRP
N__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_228612%
#average_pooling2d_1/PartitionedCall�
)res_conv_A_repr_d/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_1/PartitionedCall:output:00res_conv_a_repr_d_statefulpartitionedcall_args_10res_conv_a_repr_d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_A_repr_d_layer_call_and_return_conditional_losses_228872+
)res_conv_A_repr_d/StatefulPartitionedCall�
activation_6/PartitionedCallPartitionedCall2res_conv_A_repr_d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_6_layer_call_and_return_conditional_losses_239052
activation_6/PartitionedCall�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:04batch_normalization_6_statefulpartitionedcall_args_14batch_normalization_6_statefulpartitionedcall_args_24batch_normalization_6_statefulpartitionedcall_args_34batch_normalization_6_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_239462/
-batch_normalization_6/StatefulPartitionedCall�
)res_conv_B_repr_d/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:00res_conv_b_repr_d_statefulpartitionedcall_args_10res_conv_b_repr_d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_B_repr_d_layer_call_and_return_conditional_losses_230472+
)res_conv_B_repr_d/StatefulPartitionedCall�
activation_7/PartitionedCallPartitionedCall2res_conv_B_repr_d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_7_layer_call_and_return_conditional_losses_240002
activation_7/PartitionedCall�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:04batch_normalization_7_statefulpartitionedcall_args_14batch_normalization_7_statefulpartitionedcall_args_24batch_normalization_7_statefulpartitionedcall_args_34batch_normalization_7_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_240412/
-batch_normalization_7/StatefulPartitionedCall�
add_3/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0,average_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*I
fDRB
@__inference_add_3_layer_call_and_return_conditional_losses_240932
add_3/PartitionedCall�
"repr_board/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0)repr_board_statefulpartitionedcall_args_1)repr_board_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������*/
config_proto

GPU

CPU2 *0J 8*N
fIRG
E__inference_repr_board_layer_call_and_return_conditional_losses_232162$
"repr_board/StatefulPartitionedCall�
:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_a_repr_a_statefulpartitionedcall_args_1*^res_conv_A_repr_a/StatefulPartitionedCall*&
_output_shapes
:  *
dtype02<
:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_a/kernel/Regularizer/SquareSquareBres_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2-
+res_conv_A_repr_a/kernel/Regularizer/Square�
*res_conv_A_repr_a/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_a/kernel/Regularizer/Const�
(res_conv_A_repr_a/kernel/Regularizer/SumSum/res_conv_A_repr_a/kernel/Regularizer/Square:y:03res_conv_A_repr_a/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_a/kernel/Regularizer/Sum�
*res_conv_A_repr_a/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_a/kernel/Regularizer/mul/x�
(res_conv_A_repr_a/kernel/Regularizer/mulMul3res_conv_A_repr_a/kernel/Regularizer/mul/x:output:01res_conv_A_repr_a/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_a/kernel/Regularizer/mul�
*res_conv_A_repr_a/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_a/kernel/Regularizer/add/x�
(res_conv_A_repr_a/kernel/Regularizer/addAddV23res_conv_A_repr_a/kernel/Regularizer/add/x:output:0,res_conv_A_repr_a/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_a/kernel/Regularizer/add�
:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_b_repr_a_statefulpartitionedcall_args_1*^res_conv_B_repr_a/StatefulPartitionedCall*&
_output_shapes
:  *
dtype02<
:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_a/kernel/Regularizer/SquareSquareBres_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2-
+res_conv_B_repr_a/kernel/Regularizer/Square�
*res_conv_B_repr_a/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_a/kernel/Regularizer/Const�
(res_conv_B_repr_a/kernel/Regularizer/SumSum/res_conv_B_repr_a/kernel/Regularizer/Square:y:03res_conv_B_repr_a/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_a/kernel/Regularizer/Sum�
*res_conv_B_repr_a/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_a/kernel/Regularizer/mul/x�
(res_conv_B_repr_a/kernel/Regularizer/mulMul3res_conv_B_repr_a/kernel/Regularizer/mul/x:output:01res_conv_B_repr_a/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_a/kernel/Regularizer/mul�
*res_conv_B_repr_a/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_a/kernel/Regularizer/add/x�
(res_conv_B_repr_a/kernel/Regularizer/addAddV23res_conv_B_repr_a/kernel/Regularizer/add/x:output:0,res_conv_B_repr_a/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_a/kernel/Regularizer/add�
:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_a_repr_b_statefulpartitionedcall_args_1*^res_conv_A_repr_b/StatefulPartitionedCall*&
_output_shapes
:@@*
dtype02<
:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_b/kernel/Regularizer/SquareSquareBres_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_A_repr_b/kernel/Regularizer/Square�
*res_conv_A_repr_b/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_b/kernel/Regularizer/Const�
(res_conv_A_repr_b/kernel/Regularizer/SumSum/res_conv_A_repr_b/kernel/Regularizer/Square:y:03res_conv_A_repr_b/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_b/kernel/Regularizer/Sum�
*res_conv_A_repr_b/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_b/kernel/Regularizer/mul/x�
(res_conv_A_repr_b/kernel/Regularizer/mulMul3res_conv_A_repr_b/kernel/Regularizer/mul/x:output:01res_conv_A_repr_b/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_b/kernel/Regularizer/mul�
*res_conv_A_repr_b/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_b/kernel/Regularizer/add/x�
(res_conv_A_repr_b/kernel/Regularizer/addAddV23res_conv_A_repr_b/kernel/Regularizer/add/x:output:0,res_conv_A_repr_b/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_b/kernel/Regularizer/add�
:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_b_repr_b_statefulpartitionedcall_args_1*^res_conv_B_repr_b/StatefulPartitionedCall*&
_output_shapes
:@@*
dtype02<
:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_b/kernel/Regularizer/SquareSquareBres_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_B_repr_b/kernel/Regularizer/Square�
*res_conv_B_repr_b/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_b/kernel/Regularizer/Const�
(res_conv_B_repr_b/kernel/Regularizer/SumSum/res_conv_B_repr_b/kernel/Regularizer/Square:y:03res_conv_B_repr_b/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_b/kernel/Regularizer/Sum�
*res_conv_B_repr_b/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_b/kernel/Regularizer/mul/x�
(res_conv_B_repr_b/kernel/Regularizer/mulMul3res_conv_B_repr_b/kernel/Regularizer/mul/x:output:01res_conv_B_repr_b/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_b/kernel/Regularizer/mul�
*res_conv_B_repr_b/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_b/kernel/Regularizer/add/x�
(res_conv_B_repr_b/kernel/Regularizer/addAddV23res_conv_B_repr_b/kernel/Regularizer/add/x:output:0,res_conv_B_repr_b/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_b/kernel/Regularizer/add�
:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_a_repr_c_statefulpartitionedcall_args_1*^res_conv_A_repr_c/StatefulPartitionedCall*&
_output_shapes
:@@*
dtype02<
:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_c/kernel/Regularizer/SquareSquareBres_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_A_repr_c/kernel/Regularizer/Square�
*res_conv_A_repr_c/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_c/kernel/Regularizer/Const�
(res_conv_A_repr_c/kernel/Regularizer/SumSum/res_conv_A_repr_c/kernel/Regularizer/Square:y:03res_conv_A_repr_c/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_c/kernel/Regularizer/Sum�
*res_conv_A_repr_c/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_c/kernel/Regularizer/mul/x�
(res_conv_A_repr_c/kernel/Regularizer/mulMul3res_conv_A_repr_c/kernel/Regularizer/mul/x:output:01res_conv_A_repr_c/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_c/kernel/Regularizer/mul�
*res_conv_A_repr_c/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_c/kernel/Regularizer/add/x�
(res_conv_A_repr_c/kernel/Regularizer/addAddV23res_conv_A_repr_c/kernel/Regularizer/add/x:output:0,res_conv_A_repr_c/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_c/kernel/Regularizer/add�
:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_b_repr_c_statefulpartitionedcall_args_1*^res_conv_B_repr_c/StatefulPartitionedCall*&
_output_shapes
:@@*
dtype02<
:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_c/kernel/Regularizer/SquareSquareBres_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_B_repr_c/kernel/Regularizer/Square�
*res_conv_B_repr_c/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_c/kernel/Regularizer/Const�
(res_conv_B_repr_c/kernel/Regularizer/SumSum/res_conv_B_repr_c/kernel/Regularizer/Square:y:03res_conv_B_repr_c/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_c/kernel/Regularizer/Sum�
*res_conv_B_repr_c/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_c/kernel/Regularizer/mul/x�
(res_conv_B_repr_c/kernel/Regularizer/mulMul3res_conv_B_repr_c/kernel/Regularizer/mul/x:output:01res_conv_B_repr_c/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_c/kernel/Regularizer/mul�
*res_conv_B_repr_c/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_c/kernel/Regularizer/add/x�
(res_conv_B_repr_c/kernel/Regularizer/addAddV23res_conv_B_repr_c/kernel/Regularizer/add/x:output:0,res_conv_B_repr_c/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_c/kernel/Regularizer/add�
:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_a_repr_d_statefulpartitionedcall_args_1*^res_conv_A_repr_d/StatefulPartitionedCall*&
_output_shapes
:@@*
dtype02<
:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_d/kernel/Regularizer/SquareSquareBres_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_A_repr_d/kernel/Regularizer/Square�
*res_conv_A_repr_d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_d/kernel/Regularizer/Const�
(res_conv_A_repr_d/kernel/Regularizer/SumSum/res_conv_A_repr_d/kernel/Regularizer/Square:y:03res_conv_A_repr_d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_d/kernel/Regularizer/Sum�
*res_conv_A_repr_d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_d/kernel/Regularizer/mul/x�
(res_conv_A_repr_d/kernel/Regularizer/mulMul3res_conv_A_repr_d/kernel/Regularizer/mul/x:output:01res_conv_A_repr_d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_d/kernel/Regularizer/mul�
*res_conv_A_repr_d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_d/kernel/Regularizer/add/x�
(res_conv_A_repr_d/kernel/Regularizer/addAddV23res_conv_A_repr_d/kernel/Regularizer/add/x:output:0,res_conv_A_repr_d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_d/kernel/Regularizer/add�
:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_b_repr_d_statefulpartitionedcall_args_1*^res_conv_B_repr_d/StatefulPartitionedCall*&
_output_shapes
:@@*
dtype02<
:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_d/kernel/Regularizer/SquareSquareBres_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_B_repr_d/kernel/Regularizer/Square�
*res_conv_B_repr_d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_d/kernel/Regularizer/Const�
(res_conv_B_repr_d/kernel/Regularizer/SumSum/res_conv_B_repr_d/kernel/Regularizer/Square:y:03res_conv_B_repr_d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_d/kernel/Regularizer/Sum�
*res_conv_B_repr_d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_d/kernel/Regularizer/mul/x�
(res_conv_B_repr_d/kernel/Regularizer/mulMul3res_conv_B_repr_d/kernel/Regularizer/mul/x:output:01res_conv_B_repr_d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_d/kernel/Regularizer/mul�
*res_conv_B_repr_d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_d/kernel/Regularizer/add/x�
(res_conv_B_repr_d/kernel/Regularizer/addAddV23res_conv_B_repr_d/kernel/Regularizer/add/x:output:0,res_conv_B_repr_d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_d/kernel/Regularizer/add�
3repr_board/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)repr_board_statefulpartitionedcall_args_1#^repr_board/StatefulPartitionedCall*&
_output_shapes
:@*
dtype025
3repr_board/kernel/Regularizer/Square/ReadVariableOp�
$repr_board/kernel/Regularizer/SquareSquare;repr_board/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2&
$repr_board/kernel/Regularizer/Square�
#repr_board/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2%
#repr_board/kernel/Regularizer/Const�
!repr_board/kernel/Regularizer/SumSum(repr_board/kernel/Regularizer/Square:y:0,repr_board/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!repr_board/kernel/Regularizer/Sum�
#repr_board/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72%
#repr_board/kernel/Regularizer/mul/x�
!repr_board/kernel/Regularizer/mulMul,repr_board/kernel/Regularizer/mul/x:output:0*repr_board/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!repr_board/kernel/Regularizer/mul�
#repr_board/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#repr_board/kernel/Regularizer/add/x�
!repr_board/kernel/Regularizer/addAddV2,repr_board/kernel/Regularizer/add/x:output:0%repr_board/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2#
!repr_board/kernel/Regularizer/add�
1repr_board/bias/Regularizer/Square/ReadVariableOpReadVariableOp)repr_board_statefulpartitionedcall_args_2#^repr_board/StatefulPartitionedCall*
_output_shapes
:*
dtype023
1repr_board/bias/Regularizer/Square/ReadVariableOp�
"repr_board/bias/Regularizer/SquareSquare9repr_board/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2$
"repr_board/bias/Regularizer/Square�
!repr_board/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!repr_board/bias/Regularizer/Const�
repr_board/bias/Regularizer/SumSum&repr_board/bias/Regularizer/Square:y:0*repr_board/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
repr_board/bias/Regularizer/Sum�
!repr_board/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72#
!repr_board/bias/Regularizer/mul/x�
repr_board/bias/Regularizer/mulMul*repr_board/bias/Regularizer/mul/x:output:0(repr_board/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
repr_board/bias/Regularizer/mul�
!repr_board/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!repr_board/bias/Regularizer/add/x�
repr_board/bias/Regularizer/addAddV2*repr_board/bias/Regularizer/add/x:output:0#repr_board/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
repr_board/bias/Regularizer/add�
IdentityIdentity+repr_board/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall#^repr_board/StatefulPartitionedCall2^repr_board/bias/Regularizer/Square/ReadVariableOp4^repr_board/kernel/Regularizer/Square/ReadVariableOp*^res_conv_A_repr_a/StatefulPartitionedCall;^res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp*^res_conv_A_repr_b/StatefulPartitionedCall;^res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp*^res_conv_A_repr_c/StatefulPartitionedCall;^res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp*^res_conv_A_repr_d/StatefulPartitionedCall;^res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp*^res_conv_B_repr_a/StatefulPartitionedCall;^res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp*^res_conv_B_repr_b/StatefulPartitionedCall;^res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp*^res_conv_B_repr_c/StatefulPartitionedCall;^res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp*^res_conv_B_repr_d/StatefulPartitionedCall;^res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������``::::::::::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2H
"repr_board/StatefulPartitionedCall"repr_board/StatefulPartitionedCall2f
1repr_board/bias/Regularizer/Square/ReadVariableOp1repr_board/bias/Regularizer/Square/ReadVariableOp2j
3repr_board/kernel/Regularizer/Square/ReadVariableOp3repr_board/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_A_repr_a/StatefulPartitionedCall)res_conv_A_repr_a/StatefulPartitionedCall2x
:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_A_repr_b/StatefulPartitionedCall)res_conv_A_repr_b/StatefulPartitionedCall2x
:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_A_repr_c/StatefulPartitionedCall)res_conv_A_repr_c/StatefulPartitionedCall2x
:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_A_repr_d/StatefulPartitionedCall)res_conv_A_repr_d/StatefulPartitionedCall2x
:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_B_repr_a/StatefulPartitionedCall)res_conv_B_repr_a/StatefulPartitionedCall2x
:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_B_repr_b/StatefulPartitionedCall)res_conv_B_repr_b/StatefulPartitionedCall2x
:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_B_repr_c/StatefulPartitionedCall)res_conv_B_repr_c/StatefulPartitionedCall2x
:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_B_repr_d/StatefulPartitionedCall)res_conv_B_repr_d/StatefulPartitionedCall2x
:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_27348

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
a
E__inference_activation_layer_call_and_return_conditional_losses_25993

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������00 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������00 2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������00 :& "
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_4_layer_call_fn_26820

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_226882
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_4_layer_call_fn_26894

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_237622
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
3__inference_batch_normalization_layer_call_fn_26075

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_233262
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������00 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������00 ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_2_layer_call_fn_26443

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_223252
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_23443

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������00 : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������00 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������00 ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
H
,__inference_activation_1_layer_call_fn_26176

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_233802
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������00 2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������00 :& "
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_23857

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
c
G__inference_activation_7_layer_call_and_return_conditional_losses_24000

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
C
'__inference_reshape_layer_call_fn_25961

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:���������``*/
config_proto

GPU

CPU2 *0J 8*K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_232432
PartitionedCallx
IdentityIdentityPartitionedCall:output:0*
T0*3
_output_shapes!
:���������``2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������``:& "
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_2_layer_call_fn_26517

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_235342
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
M
1__inference_average_pooling2d_layer_call_fn_22535

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4������������������������������������*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_225292
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
H
,__inference_activation_3_layer_call_fn_26544

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_235882
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�$
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_21985

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_21970
assignmovingavg_1_21977
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/21970*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/21970*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_21970*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/21970*
_output_shapes
: 2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/21970*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_21970AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/21970*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/21977*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/21977*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_21977*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/21977*
_output_shapes
: 2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/21977*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_21977AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/21977*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
__inference_loss_fn_5_27546G
Cres_conv_b_repr_c_kernel_regularizer_square_readvariableop_resource
identity��:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp�
:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOpReadVariableOpCres_conv_b_repr_c_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@@*
dtype02<
:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_c/kernel/Regularizer/SquareSquareBres_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_B_repr_c/kernel/Regularizer/Square�
*res_conv_B_repr_c/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_c/kernel/Regularizer/Const�
(res_conv_B_repr_c/kernel/Regularizer/SumSum/res_conv_B_repr_c/kernel/Regularizer/Square:y:03res_conv_B_repr_c/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_c/kernel/Regularizer/Sum�
*res_conv_B_repr_c/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_c/kernel/Regularizer/mul/x�
(res_conv_B_repr_c/kernel/Regularizer/mulMul3res_conv_B_repr_c/kernel/Regularizer/mul/x:output:01res_conv_B_repr_c/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_c/kernel/Regularizer/mul�
*res_conv_B_repr_c/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_c/kernel/Regularizer/add/x�
(res_conv_B_repr_c/kernel/Regularizer/addAddV23res_conv_B_repr_c/kernel/Regularizer/add/x:output:0,res_conv_B_repr_c/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_c/kernel/Regularizer/add�
IdentityIdentity,res_conv_B_repr_c/kernel/Regularizer/add:z:0;^res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2x
:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp
�
c
G__inference_activation_6_layer_call_and_return_conditional_losses_27097

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
Q
%__inference_add_2_layer_call_fn_27084
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*I
fDRB
@__inference_add_2_layer_call_and_return_conditional_losses_238872
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������@:���������@:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
j
@__inference_add_1_layer_call_and_return_conditional_losses_23681

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������@:���������@:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
�
�
&__inference_conv2d_layer_call_fn_21863

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+��������������������������� */
config_proto

GPU

CPU2 *0J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_218552
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
1__inference_res_conv_A_repr_c_layer_call_fn_22563

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_A_repr_c_layer_call_and_return_conditional_losses_225552
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
1__inference_res_conv_B_repr_c_layer_call_fn_22723

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_B_repr_c_layer_call_and_return_conditional_losses_227152
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
c
G__inference_activation_3_layer_call_and_return_conditional_losses_26539

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�#
�
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_26958

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_26943
assignmovingavg_1_26950
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/26943*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/26943*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_26943*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/26943*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/26943*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_26943AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/26943*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/26950*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/26950*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_26950*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/26950*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/26950*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_26950AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/26950*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
__inference_loss_fn_9_27598>
:repr_board_bias_regularizer_square_readvariableop_resource
identity��1repr_board/bias/Regularizer/Square/ReadVariableOp�
1repr_board/bias/Regularizer/Square/ReadVariableOpReadVariableOp:repr_board_bias_regularizer_square_readvariableop_resource*
_output_shapes
:*
dtype023
1repr_board/bias/Regularizer/Square/ReadVariableOp�
"repr_board/bias/Regularizer/SquareSquare9repr_board/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2$
"repr_board/bias/Regularizer/Square�
!repr_board/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!repr_board/bias/Regularizer/Const�
repr_board/bias/Regularizer/SumSum&repr_board/bias/Regularizer/Square:y:0*repr_board/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
repr_board/bias/Regularizer/Sum�
!repr_board/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72#
!repr_board/bias/Regularizer/mul/x�
repr_board/bias/Regularizer/mulMul*repr_board/bias/Regularizer/mul/x:output:0(repr_board/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
repr_board/bias/Regularizer/mul�
!repr_board/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!repr_board/bias/Regularizer/add/x�
repr_board/bias/Regularizer/addAddV2*repr_board/bias/Regularizer/add/x:output:0#repr_board/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
repr_board/bias/Regularizer/add�
IdentityIdentity#repr_board/bias/Regularizer/add:z:02^repr_board/bias/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2f
1repr_board/bias/Regularizer/Square/ReadVariableOp1repr_board/bias/Regularizer/Square/ReadVariableOp
�
�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_26686

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�#
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_23421

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_23406
assignmovingavg_1_23413
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������00 : : : : :*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/23406*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/23406*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23406*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/23406*
_output_shapes
: 2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/23406*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23406AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/23406*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/23413*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/23413*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23413*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/23413*
_output_shapes
: 2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/23413*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23413AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/23413*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������00 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������00 ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_26612

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
H
,__inference_activation_2_layer_call_fn_26366

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_234932
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�$
�
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_27032

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_27017
assignmovingavg_1_27024
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/27017*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/27017*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_27017*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/27017*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/27017*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_27017AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/27017*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/27024*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/27024*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_27024*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/27024*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/27024*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_27024AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/27024*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_6_layer_call_fn_27179

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_229892
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
L__inference_res_conv_B_repr_a_layer_call_and_return_conditional_losses_22043

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� 2	
BiasAdd�
:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource^Conv2D/ReadVariableOp*&
_output_shapes
:  *
dtype02<
:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_a/kernel/Regularizer/SquareSquareBres_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2-
+res_conv_B_repr_a/kernel/Regularizer/Square�
*res_conv_B_repr_a/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_a/kernel/Regularizer/Const�
(res_conv_B_repr_a/kernel/Regularizer/SumSum/res_conv_B_repr_a/kernel/Regularizer/Square:y:03res_conv_B_repr_a/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_a/kernel/Regularizer/Sum�
*res_conv_B_repr_a/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_a/kernel/Regularizer/mul/x�
(res_conv_B_repr_a/kernel/Regularizer/mulMul3res_conv_B_repr_a/kernel/Regularizer/mul/x:output:01res_conv_B_repr_a/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_a/kernel/Regularizer/mul�
*res_conv_B_repr_a/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_a/kernel/Regularizer/add/x�
(res_conv_B_repr_a/kernel/Regularizer/addAddV23res_conv_B_repr_a/kernel/Regularizer/add/x:output:0,res_conv_B_repr_a/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_a/kernel/Regularizer/add�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp;^res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2x
:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs
�
Q
%__inference_add_1_layer_call_fn_26716
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_236812
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������@:���������@:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
�
__inference_loss_fn_4_27533G
Cres_conv_a_repr_c_kernel_regularizer_square_readvariableop_resource
identity��:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp�
:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOpReadVariableOpCres_conv_a_repr_c_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@@*
dtype02<
:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_c/kernel/Regularizer/SquareSquareBres_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_A_repr_c/kernel/Regularizer/Square�
*res_conv_A_repr_c/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_c/kernel/Regularizer/Const�
(res_conv_A_repr_c/kernel/Regularizer/SumSum/res_conv_A_repr_c/kernel/Regularizer/Square:y:03res_conv_A_repr_c/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_c/kernel/Regularizer/Sum�
*res_conv_A_repr_c/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_c/kernel/Regularizer/mul/x�
(res_conv_A_repr_c/kernel/Regularizer/mulMul3res_conv_A_repr_c/kernel/Regularizer/mul/x:output:01res_conv_A_repr_c/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_c/kernel/Regularizer/mul�
*res_conv_A_repr_c/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_c/kernel/Regularizer/add/x�
(res_conv_A_repr_c/kernel/Regularizer/addAddV23res_conv_A_repr_c/kernel/Regularizer/add/x:output:0,res_conv_A_repr_c/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_c/kernel/Regularizer/add�
IdentityIdentity,res_conv_A_repr_c/kernel/Regularizer/add:z:0;^res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2x
:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp
�$
�
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_22657

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_22642
assignmovingavg_1_22649
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/22642*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/22642*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_22642*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/22642*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/22642*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_22642AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/22642*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/22649*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/22649*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_22649*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/22649*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/22649*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_22649AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/22649*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
c
G__inference_activation_2_layer_call_and_return_conditional_losses_23493

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_5_layer_call_fn_26998

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_238572
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�#
�
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_27400

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_27385
assignmovingavg_1_27392
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/27385*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/27385*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_27385*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/27385*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/27385*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_27385AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/27385*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/27392*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/27392*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_27392*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/27392*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/27392*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_27392AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/27392*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_1_layer_call_fn_26327

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+��������������������������� */
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_221452
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�$
�
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_27326

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_27311
assignmovingavg_1_27318
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/27311*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/27311*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_27311*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/27311*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/27311*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_27311AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/27311*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/27318*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/27318*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_27318*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/27318*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/27318*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_27318AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/27318*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�#
�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_26590

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_26575
assignmovingavg_1_26582
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/26575*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/26575*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_26575*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/26575*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/26575*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_26575AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/26575*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/26582*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/26582*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_26582*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/26582*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/26582*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_26582AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/26582*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
__inference_loss_fn_3_27520G
Cres_conv_b_repr_b_kernel_regularizer_square_readvariableop_resource
identity��:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp�
:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOpReadVariableOpCres_conv_b_repr_b_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@@*
dtype02<
:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_b/kernel/Regularizer/SquareSquareBres_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_B_repr_b/kernel/Regularizer/Square�
*res_conv_B_repr_b/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_b/kernel/Regularizer/Const�
(res_conv_B_repr_b/kernel/Regularizer/SumSum/res_conv_B_repr_b/kernel/Regularizer/Square:y:03res_conv_B_repr_b/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_b/kernel/Regularizer/Sum�
*res_conv_B_repr_b/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_b/kernel/Regularizer/mul/x�
(res_conv_B_repr_b/kernel/Regularizer/mulMul3res_conv_B_repr_b/kernel/Regularizer/mul/x:output:01res_conv_B_repr_b/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_b/kernel/Regularizer/mul�
*res_conv_B_repr_b/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_b/kernel/Regularizer/add/x�
(res_conv_B_repr_b/kernel/Regularizer/addAddV23res_conv_B_repr_b/kernel/Regularizer/add/x:output:0,res_conv_B_repr_b/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_b/kernel/Regularizer/add�
IdentityIdentity,res_conv_B_repr_b/kernel/Regularizer/add:z:0;^res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2x
:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp
�
c
G__inference_activation_1_layer_call_and_return_conditional_losses_23380

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������00 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������00 2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������00 :& "
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_2_layer_call_fn_26526

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_235562
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�#
�
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_23835

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_23820
assignmovingavg_1_23827
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/23820*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/23820*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23820*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/23820*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/23820*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23820AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/23820*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/23827*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/23827*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23827*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/23827*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/23827*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23827AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/23827*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_7_layer_call_fn_27366

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_231802
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
L__inference_res_conv_A_repr_d_layer_call_and_return_conditional_losses_22887

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAdd�
:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource^Conv2D/ReadVariableOp*&
_output_shapes
:@@*
dtype02<
:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_d/kernel/Regularizer/SquareSquareBres_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_A_repr_d/kernel/Regularizer/Square�
*res_conv_A_repr_d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_d/kernel/Regularizer/Const�
(res_conv_A_repr_d/kernel/Regularizer/SumSum/res_conv_A_repr_d/kernel/Regularizer/Square:y:03res_conv_A_repr_d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_d/kernel/Regularizer/Sum�
*res_conv_A_repr_d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_d/kernel/Regularizer/mul/x�
(res_conv_A_repr_d/kernel/Regularizer/mulMul3res_conv_A_repr_d/kernel/Regularizer/mul/x:output:01res_conv_A_repr_d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_d/kernel/Regularizer/mul�
*res_conv_A_repr_d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_d/kernel/Regularizer/add/x�
(res_conv_A_repr_d/kernel/Regularizer/addAddV23res_conv_A_repr_d/kernel/Regularizer/add/x:output:0,res_conv_A_repr_d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_d/kernel/Regularizer/add�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp;^res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2x
:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs
��
�8
 __inference__wrapped_model_21830	
board8
4representation_conv2d_conv2d_readvariableop_resource9
5representation_conv2d_biasadd_readvariableop_resourceC
?representation_res_conv_a_repr_a_conv2d_readvariableop_resourceD
@representation_res_conv_a_repr_a_biasadd_readvariableop_resource>
:representation_batch_normalization_readvariableop_resource@
<representation_batch_normalization_readvariableop_1_resourceO
Krepresentation_batch_normalization_fusedbatchnormv3_readvariableop_resourceQ
Mrepresentation_batch_normalization_fusedbatchnormv3_readvariableop_1_resourceC
?representation_res_conv_b_repr_a_conv2d_readvariableop_resourceD
@representation_res_conv_b_repr_a_biasadd_readvariableop_resource@
<representation_batch_normalization_1_readvariableop_resourceB
>representation_batch_normalization_1_readvariableop_1_resourceQ
Mrepresentation_batch_normalization_1_fusedbatchnormv3_readvariableop_resourceS
Orepresentation_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource:
6representation_conv2d_1_conv2d_readvariableop_resource;
7representation_conv2d_1_biasadd_readvariableop_resourceC
?representation_res_conv_a_repr_b_conv2d_readvariableop_resourceD
@representation_res_conv_a_repr_b_biasadd_readvariableop_resource@
<representation_batch_normalization_2_readvariableop_resourceB
>representation_batch_normalization_2_readvariableop_1_resourceQ
Mrepresentation_batch_normalization_2_fusedbatchnormv3_readvariableop_resourceS
Orepresentation_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resourceC
?representation_res_conv_b_repr_b_conv2d_readvariableop_resourceD
@representation_res_conv_b_repr_b_biasadd_readvariableop_resource@
<representation_batch_normalization_3_readvariableop_resourceB
>representation_batch_normalization_3_readvariableop_1_resourceQ
Mrepresentation_batch_normalization_3_fusedbatchnormv3_readvariableop_resourceS
Orepresentation_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resourceC
?representation_res_conv_a_repr_c_conv2d_readvariableop_resourceD
@representation_res_conv_a_repr_c_biasadd_readvariableop_resource@
<representation_batch_normalization_4_readvariableop_resourceB
>representation_batch_normalization_4_readvariableop_1_resourceQ
Mrepresentation_batch_normalization_4_fusedbatchnormv3_readvariableop_resourceS
Orepresentation_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resourceC
?representation_res_conv_b_repr_c_conv2d_readvariableop_resourceD
@representation_res_conv_b_repr_c_biasadd_readvariableop_resource@
<representation_batch_normalization_5_readvariableop_resourceB
>representation_batch_normalization_5_readvariableop_1_resourceQ
Mrepresentation_batch_normalization_5_fusedbatchnormv3_readvariableop_resourceS
Orepresentation_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resourceC
?representation_res_conv_a_repr_d_conv2d_readvariableop_resourceD
@representation_res_conv_a_repr_d_biasadd_readvariableop_resource@
<representation_batch_normalization_6_readvariableop_resourceB
>representation_batch_normalization_6_readvariableop_1_resourceQ
Mrepresentation_batch_normalization_6_fusedbatchnormv3_readvariableop_resourceS
Orepresentation_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resourceC
?representation_res_conv_b_repr_d_conv2d_readvariableop_resourceD
@representation_res_conv_b_repr_d_biasadd_readvariableop_resource@
<representation_batch_normalization_7_readvariableop_resourceB
>representation_batch_normalization_7_readvariableop_1_resourceQ
Mrepresentation_batch_normalization_7_fusedbatchnormv3_readvariableop_resourceS
Orepresentation_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource<
8representation_repr_board_conv2d_readvariableop_resource=
9representation_repr_board_biasadd_readvariableop_resource
identity��BRepresentation/batch_normalization/FusedBatchNormV3/ReadVariableOp�DRepresentation/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�1Representation/batch_normalization/ReadVariableOp�3Representation/batch_normalization/ReadVariableOp_1�DRepresentation/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�FRepresentation/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�3Representation/batch_normalization_1/ReadVariableOp�5Representation/batch_normalization_1/ReadVariableOp_1�DRepresentation/batch_normalization_2/FusedBatchNormV3/ReadVariableOp�FRepresentation/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�3Representation/batch_normalization_2/ReadVariableOp�5Representation/batch_normalization_2/ReadVariableOp_1�DRepresentation/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�FRepresentation/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�3Representation/batch_normalization_3/ReadVariableOp�5Representation/batch_normalization_3/ReadVariableOp_1�DRepresentation/batch_normalization_4/FusedBatchNormV3/ReadVariableOp�FRepresentation/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�3Representation/batch_normalization_4/ReadVariableOp�5Representation/batch_normalization_4/ReadVariableOp_1�DRepresentation/batch_normalization_5/FusedBatchNormV3/ReadVariableOp�FRepresentation/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�3Representation/batch_normalization_5/ReadVariableOp�5Representation/batch_normalization_5/ReadVariableOp_1�DRepresentation/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�FRepresentation/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�3Representation/batch_normalization_6/ReadVariableOp�5Representation/batch_normalization_6/ReadVariableOp_1�DRepresentation/batch_normalization_7/FusedBatchNormV3/ReadVariableOp�FRepresentation/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1�3Representation/batch_normalization_7/ReadVariableOp�5Representation/batch_normalization_7/ReadVariableOp_1�,Representation/conv2d/BiasAdd/ReadVariableOp�+Representation/conv2d/Conv2D/ReadVariableOp�.Representation/conv2d_1/BiasAdd/ReadVariableOp�-Representation/conv2d_1/Conv2D/ReadVariableOp�0Representation/repr_board/BiasAdd/ReadVariableOp�/Representation/repr_board/Conv2D/ReadVariableOp�7Representation/res_conv_A_repr_a/BiasAdd/ReadVariableOp�6Representation/res_conv_A_repr_a/Conv2D/ReadVariableOp�7Representation/res_conv_A_repr_b/BiasAdd/ReadVariableOp�6Representation/res_conv_A_repr_b/Conv2D/ReadVariableOp�7Representation/res_conv_A_repr_c/BiasAdd/ReadVariableOp�6Representation/res_conv_A_repr_c/Conv2D/ReadVariableOp�7Representation/res_conv_A_repr_d/BiasAdd/ReadVariableOp�6Representation/res_conv_A_repr_d/Conv2D/ReadVariableOp�7Representation/res_conv_B_repr_a/BiasAdd/ReadVariableOp�6Representation/res_conv_B_repr_a/Conv2D/ReadVariableOp�7Representation/res_conv_B_repr_b/BiasAdd/ReadVariableOp�6Representation/res_conv_B_repr_b/Conv2D/ReadVariableOp�7Representation/res_conv_B_repr_c/BiasAdd/ReadVariableOp�6Representation/res_conv_B_repr_c/Conv2D/ReadVariableOp�7Representation/res_conv_B_repr_d/BiasAdd/ReadVariableOp�6Representation/res_conv_B_repr_d/Conv2D/ReadVariableOpq
Representation/reshape/ShapeShapeboard*
T0*
_output_shapes
:2
Representation/reshape/Shape�
*Representation/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*Representation/reshape/strided_slice/stack�
,Representation/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,Representation/reshape/strided_slice/stack_1�
,Representation/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Representation/reshape/strided_slice/stack_2�
$Representation/reshape/strided_sliceStridedSlice%Representation/reshape/Shape:output:03Representation/reshape/strided_slice/stack:output:05Representation/reshape/strided_slice/stack_1:output:05Representation/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$Representation/reshape/strided_slice�
&Representation/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2(
&Representation/reshape/Reshape/shape/1�
&Representation/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :`2(
&Representation/reshape/Reshape/shape/2�
&Representation/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :`2(
&Representation/reshape/Reshape/shape/3�
&Representation/reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2(
&Representation/reshape/Reshape/shape/4�
$Representation/reshape/Reshape/shapePack-Representation/reshape/strided_slice:output:0/Representation/reshape/Reshape/shape/1:output:0/Representation/reshape/Reshape/shape/2:output:0/Representation/reshape/Reshape/shape/3:output:0/Representation/reshape/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2&
$Representation/reshape/Reshape/shape�
Representation/reshape/ReshapeReshapeboard-Representation/reshape/Reshape/shape:output:0*
T0*3
_output_shapes!
:���������``2 
Representation/reshape/Reshape�
%Representation/permute/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2'
%Representation/permute/transpose/perm�
 Representation/permute/transpose	Transpose'Representation/reshape/Reshape:output:0.Representation/permute/transpose/perm:output:0*
T0*3
_output_shapes!
:���������``2"
 Representation/permute/transpose�
Representation/reshape_1/ShapeShape$Representation/permute/transpose:y:0*
T0*
_output_shapes
:2 
Representation/reshape_1/Shape�
,Representation/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,Representation/reshape_1/strided_slice/stack�
.Representation/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.Representation/reshape_1/strided_slice/stack_1�
.Representation/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.Representation/reshape_1/strided_slice/stack_2�
&Representation/reshape_1/strided_sliceStridedSlice'Representation/reshape_1/Shape:output:05Representation/reshape_1/strided_slice/stack:output:07Representation/reshape_1/strided_slice/stack_1:output:07Representation/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&Representation/reshape_1/strided_slice�
(Representation/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :`2*
(Representation/reshape_1/Reshape/shape/1�
(Representation/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :`2*
(Representation/reshape_1/Reshape/shape/2�
(Representation/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2*
(Representation/reshape_1/Reshape/shape/3�
&Representation/reshape_1/Reshape/shapePack/Representation/reshape_1/strided_slice:output:01Representation/reshape_1/Reshape/shape/1:output:01Representation/reshape_1/Reshape/shape/2:output:01Representation/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2(
&Representation/reshape_1/Reshape/shape�
 Representation/reshape_1/ReshapeReshape$Representation/permute/transpose:y:0/Representation/reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������``2"
 Representation/reshape_1/Reshape�
+Representation/conv2d/Conv2D/ReadVariableOpReadVariableOp4representation_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+Representation/conv2d/Conv2D/ReadVariableOp�
Representation/conv2d/Conv2DConv2D)Representation/reshape_1/Reshape:output:03Representation/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00 *
paddingSAME*
strides
2
Representation/conv2d/Conv2D�
,Representation/conv2d/BiasAdd/ReadVariableOpReadVariableOp5representation_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02.
,Representation/conv2d/BiasAdd/ReadVariableOp�
Representation/conv2d/BiasAddBiasAdd%Representation/conv2d/Conv2D:output:04Representation/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00 2
Representation/conv2d/BiasAdd�
6Representation/res_conv_A_repr_a/Conv2D/ReadVariableOpReadVariableOp?representation_res_conv_a_repr_a_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype028
6Representation/res_conv_A_repr_a/Conv2D/ReadVariableOp�
'Representation/res_conv_A_repr_a/Conv2DConv2D&Representation/conv2d/BiasAdd:output:0>Representation/res_conv_A_repr_a/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00 *
paddingSAME*
strides
2)
'Representation/res_conv_A_repr_a/Conv2D�
7Representation/res_conv_A_repr_a/BiasAdd/ReadVariableOpReadVariableOp@representation_res_conv_a_repr_a_biasadd_readvariableop_resource*
_output_shapes
: *
dtype029
7Representation/res_conv_A_repr_a/BiasAdd/ReadVariableOp�
(Representation/res_conv_A_repr_a/BiasAddBiasAdd0Representation/res_conv_A_repr_a/Conv2D:output:0?Representation/res_conv_A_repr_a/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00 2*
(Representation/res_conv_A_repr_a/BiasAdd�
Representation/activation/ReluRelu1Representation/res_conv_A_repr_a/BiasAdd:output:0*
T0*/
_output_shapes
:���������00 2 
Representation/activation/Relu�
/Representation/batch_normalization/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 21
/Representation/batch_normalization/LogicalAnd/x�
/Representation/batch_normalization/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z21
/Representation/batch_normalization/LogicalAnd/y�
-Representation/batch_normalization/LogicalAnd
LogicalAnd8Representation/batch_normalization/LogicalAnd/x:output:08Representation/batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: 2/
-Representation/batch_normalization/LogicalAnd�
1Representation/batch_normalization/ReadVariableOpReadVariableOp:representation_batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype023
1Representation/batch_normalization/ReadVariableOp�
3Representation/batch_normalization/ReadVariableOp_1ReadVariableOp<representation_batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype025
3Representation/batch_normalization/ReadVariableOp_1�
BRepresentation/batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOpKrepresentation_batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02D
BRepresentation/batch_normalization/FusedBatchNormV3/ReadVariableOp�
DRepresentation/batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpMrepresentation_batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02F
DRepresentation/batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
3Representation/batch_normalization/FusedBatchNormV3FusedBatchNormV3,Representation/activation/Relu:activations:09Representation/batch_normalization/ReadVariableOp:value:0;Representation/batch_normalization/ReadVariableOp_1:value:0JRepresentation/batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0LRepresentation/batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������00 : : : : :*
epsilon%o�:*
is_training( 25
3Representation/batch_normalization/FusedBatchNormV3�
(Representation/batch_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2*
(Representation/batch_normalization/Const�
6Representation/res_conv_B_repr_a/Conv2D/ReadVariableOpReadVariableOp?representation_res_conv_b_repr_a_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype028
6Representation/res_conv_B_repr_a/Conv2D/ReadVariableOp�
'Representation/res_conv_B_repr_a/Conv2DConv2D7Representation/batch_normalization/FusedBatchNormV3:y:0>Representation/res_conv_B_repr_a/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00 *
paddingSAME*
strides
2)
'Representation/res_conv_B_repr_a/Conv2D�
7Representation/res_conv_B_repr_a/BiasAdd/ReadVariableOpReadVariableOp@representation_res_conv_b_repr_a_biasadd_readvariableop_resource*
_output_shapes
: *
dtype029
7Representation/res_conv_B_repr_a/BiasAdd/ReadVariableOp�
(Representation/res_conv_B_repr_a/BiasAddBiasAdd0Representation/res_conv_B_repr_a/Conv2D:output:0?Representation/res_conv_B_repr_a/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00 2*
(Representation/res_conv_B_repr_a/BiasAdd�
 Representation/activation_1/ReluRelu1Representation/res_conv_B_repr_a/BiasAdd:output:0*
T0*/
_output_shapes
:���������00 2"
 Representation/activation_1/Relu�
1Representation/batch_normalization_1/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 23
1Representation/batch_normalization_1/LogicalAnd/x�
1Representation/batch_normalization_1/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z23
1Representation/batch_normalization_1/LogicalAnd/y�
/Representation/batch_normalization_1/LogicalAnd
LogicalAnd:Representation/batch_normalization_1/LogicalAnd/x:output:0:Representation/batch_normalization_1/LogicalAnd/y:output:0*
_output_shapes
: 21
/Representation/batch_normalization_1/LogicalAnd�
3Representation/batch_normalization_1/ReadVariableOpReadVariableOp<representation_batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype025
3Representation/batch_normalization_1/ReadVariableOp�
5Representation/batch_normalization_1/ReadVariableOp_1ReadVariableOp>representation_batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype027
5Representation/batch_normalization_1/ReadVariableOp_1�
DRepresentation/batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOpMrepresentation_batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02F
DRepresentation/batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
FRepresentation/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOrepresentation_batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02H
FRepresentation/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
5Representation/batch_normalization_1/FusedBatchNormV3FusedBatchNormV3.Representation/activation_1/Relu:activations:0;Representation/batch_normalization_1/ReadVariableOp:value:0=Representation/batch_normalization_1/ReadVariableOp_1:value:0LRepresentation/batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0NRepresentation/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������00 : : : : :*
epsilon%o�:*
is_training( 27
5Representation/batch_normalization_1/FusedBatchNormV3�
*Representation/batch_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2,
*Representation/batch_normalization_1/Const�
Representation/add/addAddV29Representation/batch_normalization_1/FusedBatchNormV3:y:0&Representation/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������00 2
Representation/add/add�
-Representation/conv2d_1/Conv2D/ReadVariableOpReadVariableOp6representation_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02/
-Representation/conv2d_1/Conv2D/ReadVariableOp�
Representation/conv2d_1/Conv2DConv2DRepresentation/add/add:z:05Representation/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2 
Representation/conv2d_1/Conv2D�
.Representation/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp7representation_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype020
.Representation/conv2d_1/BiasAdd/ReadVariableOp�
Representation/conv2d_1/BiasAddBiasAdd'Representation/conv2d_1/Conv2D:output:06Representation/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2!
Representation/conv2d_1/BiasAdd�
6Representation/res_conv_A_repr_b/Conv2D/ReadVariableOpReadVariableOp?representation_res_conv_a_repr_b_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype028
6Representation/res_conv_A_repr_b/Conv2D/ReadVariableOp�
'Representation/res_conv_A_repr_b/Conv2DConv2D(Representation/conv2d_1/BiasAdd:output:0>Representation/res_conv_A_repr_b/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2)
'Representation/res_conv_A_repr_b/Conv2D�
7Representation/res_conv_A_repr_b/BiasAdd/ReadVariableOpReadVariableOp@representation_res_conv_a_repr_b_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7Representation/res_conv_A_repr_b/BiasAdd/ReadVariableOp�
(Representation/res_conv_A_repr_b/BiasAddBiasAdd0Representation/res_conv_A_repr_b/Conv2D:output:0?Representation/res_conv_A_repr_b/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2*
(Representation/res_conv_A_repr_b/BiasAdd�
 Representation/activation_2/ReluRelu1Representation/res_conv_A_repr_b/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2"
 Representation/activation_2/Relu�
1Representation/batch_normalization_2/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 23
1Representation/batch_normalization_2/LogicalAnd/x�
1Representation/batch_normalization_2/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z23
1Representation/batch_normalization_2/LogicalAnd/y�
/Representation/batch_normalization_2/LogicalAnd
LogicalAnd:Representation/batch_normalization_2/LogicalAnd/x:output:0:Representation/batch_normalization_2/LogicalAnd/y:output:0*
_output_shapes
: 21
/Representation/batch_normalization_2/LogicalAnd�
3Representation/batch_normalization_2/ReadVariableOpReadVariableOp<representation_batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype025
3Representation/batch_normalization_2/ReadVariableOp�
5Representation/batch_normalization_2/ReadVariableOp_1ReadVariableOp>representation_batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype027
5Representation/batch_normalization_2/ReadVariableOp_1�
DRepresentation/batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOpMrepresentation_batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02F
DRepresentation/batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
FRepresentation/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOrepresentation_batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02H
FRepresentation/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
5Representation/batch_normalization_2/FusedBatchNormV3FusedBatchNormV3.Representation/activation_2/Relu:activations:0;Representation/batch_normalization_2/ReadVariableOp:value:0=Representation/batch_normalization_2/ReadVariableOp_1:value:0LRepresentation/batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0NRepresentation/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 27
5Representation/batch_normalization_2/FusedBatchNormV3�
*Representation/batch_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2,
*Representation/batch_normalization_2/Const�
6Representation/res_conv_B_repr_b/Conv2D/ReadVariableOpReadVariableOp?representation_res_conv_b_repr_b_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype028
6Representation/res_conv_B_repr_b/Conv2D/ReadVariableOp�
'Representation/res_conv_B_repr_b/Conv2DConv2D9Representation/batch_normalization_2/FusedBatchNormV3:y:0>Representation/res_conv_B_repr_b/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2)
'Representation/res_conv_B_repr_b/Conv2D�
7Representation/res_conv_B_repr_b/BiasAdd/ReadVariableOpReadVariableOp@representation_res_conv_b_repr_b_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7Representation/res_conv_B_repr_b/BiasAdd/ReadVariableOp�
(Representation/res_conv_B_repr_b/BiasAddBiasAdd0Representation/res_conv_B_repr_b/Conv2D:output:0?Representation/res_conv_B_repr_b/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2*
(Representation/res_conv_B_repr_b/BiasAdd�
 Representation/activation_3/ReluRelu1Representation/res_conv_B_repr_b/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2"
 Representation/activation_3/Relu�
1Representation/batch_normalization_3/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 23
1Representation/batch_normalization_3/LogicalAnd/x�
1Representation/batch_normalization_3/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z23
1Representation/batch_normalization_3/LogicalAnd/y�
/Representation/batch_normalization_3/LogicalAnd
LogicalAnd:Representation/batch_normalization_3/LogicalAnd/x:output:0:Representation/batch_normalization_3/LogicalAnd/y:output:0*
_output_shapes
: 21
/Representation/batch_normalization_3/LogicalAnd�
3Representation/batch_normalization_3/ReadVariableOpReadVariableOp<representation_batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype025
3Representation/batch_normalization_3/ReadVariableOp�
5Representation/batch_normalization_3/ReadVariableOp_1ReadVariableOp>representation_batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype027
5Representation/batch_normalization_3/ReadVariableOp_1�
DRepresentation/batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOpMrepresentation_batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02F
DRepresentation/batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
FRepresentation/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOrepresentation_batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02H
FRepresentation/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
5Representation/batch_normalization_3/FusedBatchNormV3FusedBatchNormV3.Representation/activation_3/Relu:activations:0;Representation/batch_normalization_3/ReadVariableOp:value:0=Representation/batch_normalization_3/ReadVariableOp_1:value:0LRepresentation/batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0NRepresentation/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 27
5Representation/batch_normalization_3/FusedBatchNormV3�
*Representation/batch_normalization_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2,
*Representation/batch_normalization_3/Const�
Representation/add_1/addAddV29Representation/batch_normalization_3/FusedBatchNormV3:y:0(Representation/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
Representation/add_1/add�
(Representation/average_pooling2d/AvgPoolAvgPoolRepresentation/add_1/add:z:0*
T0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2*
(Representation/average_pooling2d/AvgPool�
6Representation/res_conv_A_repr_c/Conv2D/ReadVariableOpReadVariableOp?representation_res_conv_a_repr_c_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype028
6Representation/res_conv_A_repr_c/Conv2D/ReadVariableOp�
'Representation/res_conv_A_repr_c/Conv2DConv2D1Representation/average_pooling2d/AvgPool:output:0>Representation/res_conv_A_repr_c/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2)
'Representation/res_conv_A_repr_c/Conv2D�
7Representation/res_conv_A_repr_c/BiasAdd/ReadVariableOpReadVariableOp@representation_res_conv_a_repr_c_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7Representation/res_conv_A_repr_c/BiasAdd/ReadVariableOp�
(Representation/res_conv_A_repr_c/BiasAddBiasAdd0Representation/res_conv_A_repr_c/Conv2D:output:0?Representation/res_conv_A_repr_c/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2*
(Representation/res_conv_A_repr_c/BiasAdd�
 Representation/activation_4/ReluRelu1Representation/res_conv_A_repr_c/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2"
 Representation/activation_4/Relu�
1Representation/batch_normalization_4/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 23
1Representation/batch_normalization_4/LogicalAnd/x�
1Representation/batch_normalization_4/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z23
1Representation/batch_normalization_4/LogicalAnd/y�
/Representation/batch_normalization_4/LogicalAnd
LogicalAnd:Representation/batch_normalization_4/LogicalAnd/x:output:0:Representation/batch_normalization_4/LogicalAnd/y:output:0*
_output_shapes
: 21
/Representation/batch_normalization_4/LogicalAnd�
3Representation/batch_normalization_4/ReadVariableOpReadVariableOp<representation_batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype025
3Representation/batch_normalization_4/ReadVariableOp�
5Representation/batch_normalization_4/ReadVariableOp_1ReadVariableOp>representation_batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype027
5Representation/batch_normalization_4/ReadVariableOp_1�
DRepresentation/batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOpMrepresentation_batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02F
DRepresentation/batch_normalization_4/FusedBatchNormV3/ReadVariableOp�
FRepresentation/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOrepresentation_batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02H
FRepresentation/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�
5Representation/batch_normalization_4/FusedBatchNormV3FusedBatchNormV3.Representation/activation_4/Relu:activations:0;Representation/batch_normalization_4/ReadVariableOp:value:0=Representation/batch_normalization_4/ReadVariableOp_1:value:0LRepresentation/batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0NRepresentation/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 27
5Representation/batch_normalization_4/FusedBatchNormV3�
*Representation/batch_normalization_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2,
*Representation/batch_normalization_4/Const�
6Representation/res_conv_B_repr_c/Conv2D/ReadVariableOpReadVariableOp?representation_res_conv_b_repr_c_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype028
6Representation/res_conv_B_repr_c/Conv2D/ReadVariableOp�
'Representation/res_conv_B_repr_c/Conv2DConv2D9Representation/batch_normalization_4/FusedBatchNormV3:y:0>Representation/res_conv_B_repr_c/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2)
'Representation/res_conv_B_repr_c/Conv2D�
7Representation/res_conv_B_repr_c/BiasAdd/ReadVariableOpReadVariableOp@representation_res_conv_b_repr_c_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7Representation/res_conv_B_repr_c/BiasAdd/ReadVariableOp�
(Representation/res_conv_B_repr_c/BiasAddBiasAdd0Representation/res_conv_B_repr_c/Conv2D:output:0?Representation/res_conv_B_repr_c/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2*
(Representation/res_conv_B_repr_c/BiasAdd�
 Representation/activation_5/ReluRelu1Representation/res_conv_B_repr_c/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2"
 Representation/activation_5/Relu�
1Representation/batch_normalization_5/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 23
1Representation/batch_normalization_5/LogicalAnd/x�
1Representation/batch_normalization_5/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z23
1Representation/batch_normalization_5/LogicalAnd/y�
/Representation/batch_normalization_5/LogicalAnd
LogicalAnd:Representation/batch_normalization_5/LogicalAnd/x:output:0:Representation/batch_normalization_5/LogicalAnd/y:output:0*
_output_shapes
: 21
/Representation/batch_normalization_5/LogicalAnd�
3Representation/batch_normalization_5/ReadVariableOpReadVariableOp<representation_batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype025
3Representation/batch_normalization_5/ReadVariableOp�
5Representation/batch_normalization_5/ReadVariableOp_1ReadVariableOp>representation_batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype027
5Representation/batch_normalization_5/ReadVariableOp_1�
DRepresentation/batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOpMrepresentation_batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02F
DRepresentation/batch_normalization_5/FusedBatchNormV3/ReadVariableOp�
FRepresentation/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOrepresentation_batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02H
FRepresentation/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�
5Representation/batch_normalization_5/FusedBatchNormV3FusedBatchNormV3.Representation/activation_5/Relu:activations:0;Representation/batch_normalization_5/ReadVariableOp:value:0=Representation/batch_normalization_5/ReadVariableOp_1:value:0LRepresentation/batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0NRepresentation/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 27
5Representation/batch_normalization_5/FusedBatchNormV3�
*Representation/batch_normalization_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2,
*Representation/batch_normalization_5/Const�
Representation/add_2/addAddV29Representation/batch_normalization_5/FusedBatchNormV3:y:01Representation/average_pooling2d/AvgPool:output:0*
T0*/
_output_shapes
:���������@2
Representation/add_2/add�
*Representation/average_pooling2d_1/AvgPoolAvgPoolRepresentation/add_2/add:z:0*
T0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2,
*Representation/average_pooling2d_1/AvgPool�
6Representation/res_conv_A_repr_d/Conv2D/ReadVariableOpReadVariableOp?representation_res_conv_a_repr_d_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype028
6Representation/res_conv_A_repr_d/Conv2D/ReadVariableOp�
'Representation/res_conv_A_repr_d/Conv2DConv2D3Representation/average_pooling2d_1/AvgPool:output:0>Representation/res_conv_A_repr_d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2)
'Representation/res_conv_A_repr_d/Conv2D�
7Representation/res_conv_A_repr_d/BiasAdd/ReadVariableOpReadVariableOp@representation_res_conv_a_repr_d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7Representation/res_conv_A_repr_d/BiasAdd/ReadVariableOp�
(Representation/res_conv_A_repr_d/BiasAddBiasAdd0Representation/res_conv_A_repr_d/Conv2D:output:0?Representation/res_conv_A_repr_d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2*
(Representation/res_conv_A_repr_d/BiasAdd�
 Representation/activation_6/ReluRelu1Representation/res_conv_A_repr_d/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2"
 Representation/activation_6/Relu�
1Representation/batch_normalization_6/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 23
1Representation/batch_normalization_6/LogicalAnd/x�
1Representation/batch_normalization_6/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z23
1Representation/batch_normalization_6/LogicalAnd/y�
/Representation/batch_normalization_6/LogicalAnd
LogicalAnd:Representation/batch_normalization_6/LogicalAnd/x:output:0:Representation/batch_normalization_6/LogicalAnd/y:output:0*
_output_shapes
: 21
/Representation/batch_normalization_6/LogicalAnd�
3Representation/batch_normalization_6/ReadVariableOpReadVariableOp<representation_batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype025
3Representation/batch_normalization_6/ReadVariableOp�
5Representation/batch_normalization_6/ReadVariableOp_1ReadVariableOp>representation_batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype027
5Representation/batch_normalization_6/ReadVariableOp_1�
DRepresentation/batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOpMrepresentation_batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02F
DRepresentation/batch_normalization_6/FusedBatchNormV3/ReadVariableOp�
FRepresentation/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOrepresentation_batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02H
FRepresentation/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�
5Representation/batch_normalization_6/FusedBatchNormV3FusedBatchNormV3.Representation/activation_6/Relu:activations:0;Representation/batch_normalization_6/ReadVariableOp:value:0=Representation/batch_normalization_6/ReadVariableOp_1:value:0LRepresentation/batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0NRepresentation/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 27
5Representation/batch_normalization_6/FusedBatchNormV3�
*Representation/batch_normalization_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2,
*Representation/batch_normalization_6/Const�
6Representation/res_conv_B_repr_d/Conv2D/ReadVariableOpReadVariableOp?representation_res_conv_b_repr_d_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype028
6Representation/res_conv_B_repr_d/Conv2D/ReadVariableOp�
'Representation/res_conv_B_repr_d/Conv2DConv2D9Representation/batch_normalization_6/FusedBatchNormV3:y:0>Representation/res_conv_B_repr_d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2)
'Representation/res_conv_B_repr_d/Conv2D�
7Representation/res_conv_B_repr_d/BiasAdd/ReadVariableOpReadVariableOp@representation_res_conv_b_repr_d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype029
7Representation/res_conv_B_repr_d/BiasAdd/ReadVariableOp�
(Representation/res_conv_B_repr_d/BiasAddBiasAdd0Representation/res_conv_B_repr_d/Conv2D:output:0?Representation/res_conv_B_repr_d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2*
(Representation/res_conv_B_repr_d/BiasAdd�
 Representation/activation_7/ReluRelu1Representation/res_conv_B_repr_d/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2"
 Representation/activation_7/Relu�
1Representation/batch_normalization_7/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 23
1Representation/batch_normalization_7/LogicalAnd/x�
1Representation/batch_normalization_7/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z23
1Representation/batch_normalization_7/LogicalAnd/y�
/Representation/batch_normalization_7/LogicalAnd
LogicalAnd:Representation/batch_normalization_7/LogicalAnd/x:output:0:Representation/batch_normalization_7/LogicalAnd/y:output:0*
_output_shapes
: 21
/Representation/batch_normalization_7/LogicalAnd�
3Representation/batch_normalization_7/ReadVariableOpReadVariableOp<representation_batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype025
3Representation/batch_normalization_7/ReadVariableOp�
5Representation/batch_normalization_7/ReadVariableOp_1ReadVariableOp>representation_batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype027
5Representation/batch_normalization_7/ReadVariableOp_1�
DRepresentation/batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOpMrepresentation_batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02F
DRepresentation/batch_normalization_7/FusedBatchNormV3/ReadVariableOp�
FRepresentation/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOrepresentation_batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02H
FRepresentation/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1�
5Representation/batch_normalization_7/FusedBatchNormV3FusedBatchNormV3.Representation/activation_7/Relu:activations:0;Representation/batch_normalization_7/ReadVariableOp:value:0=Representation/batch_normalization_7/ReadVariableOp_1:value:0LRepresentation/batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0NRepresentation/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 27
5Representation/batch_normalization_7/FusedBatchNormV3�
*Representation/batch_normalization_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2,
*Representation/batch_normalization_7/Const�
Representation/add_3/addAddV29Representation/batch_normalization_7/FusedBatchNormV3:y:03Representation/average_pooling2d_1/AvgPool:output:0*
T0*/
_output_shapes
:���������@2
Representation/add_3/add�
/Representation/repr_board/Conv2D/ReadVariableOpReadVariableOp8representation_repr_board_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype021
/Representation/repr_board/Conv2D/ReadVariableOp�
 Representation/repr_board/Conv2DConv2DRepresentation/add_3/add:z:07Representation/repr_board/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
2"
 Representation/repr_board/Conv2D�
0Representation/repr_board/BiasAdd/ReadVariableOpReadVariableOp9representation_repr_board_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0Representation/repr_board/BiasAdd/ReadVariableOp�
!Representation/repr_board/BiasAddBiasAdd)Representation/repr_board/Conv2D:output:08Representation/repr_board/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2#
!Representation/repr_board/BiasAdd�
Representation/repr_board/ReluRelu*Representation/repr_board/BiasAdd:output:0*
T0*/
_output_shapes
:���������2 
Representation/repr_board/Relu�
IdentityIdentity,Representation/repr_board/Relu:activations:0C^Representation/batch_normalization/FusedBatchNormV3/ReadVariableOpE^Representation/batch_normalization/FusedBatchNormV3/ReadVariableOp_12^Representation/batch_normalization/ReadVariableOp4^Representation/batch_normalization/ReadVariableOp_1E^Representation/batch_normalization_1/FusedBatchNormV3/ReadVariableOpG^Representation/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_14^Representation/batch_normalization_1/ReadVariableOp6^Representation/batch_normalization_1/ReadVariableOp_1E^Representation/batch_normalization_2/FusedBatchNormV3/ReadVariableOpG^Representation/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_14^Representation/batch_normalization_2/ReadVariableOp6^Representation/batch_normalization_2/ReadVariableOp_1E^Representation/batch_normalization_3/FusedBatchNormV3/ReadVariableOpG^Representation/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_14^Representation/batch_normalization_3/ReadVariableOp6^Representation/batch_normalization_3/ReadVariableOp_1E^Representation/batch_normalization_4/FusedBatchNormV3/ReadVariableOpG^Representation/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_14^Representation/batch_normalization_4/ReadVariableOp6^Representation/batch_normalization_4/ReadVariableOp_1E^Representation/batch_normalization_5/FusedBatchNormV3/ReadVariableOpG^Representation/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_14^Representation/batch_normalization_5/ReadVariableOp6^Representation/batch_normalization_5/ReadVariableOp_1E^Representation/batch_normalization_6/FusedBatchNormV3/ReadVariableOpG^Representation/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_14^Representation/batch_normalization_6/ReadVariableOp6^Representation/batch_normalization_6/ReadVariableOp_1E^Representation/batch_normalization_7/FusedBatchNormV3/ReadVariableOpG^Representation/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_14^Representation/batch_normalization_7/ReadVariableOp6^Representation/batch_normalization_7/ReadVariableOp_1-^Representation/conv2d/BiasAdd/ReadVariableOp,^Representation/conv2d/Conv2D/ReadVariableOp/^Representation/conv2d_1/BiasAdd/ReadVariableOp.^Representation/conv2d_1/Conv2D/ReadVariableOp1^Representation/repr_board/BiasAdd/ReadVariableOp0^Representation/repr_board/Conv2D/ReadVariableOp8^Representation/res_conv_A_repr_a/BiasAdd/ReadVariableOp7^Representation/res_conv_A_repr_a/Conv2D/ReadVariableOp8^Representation/res_conv_A_repr_b/BiasAdd/ReadVariableOp7^Representation/res_conv_A_repr_b/Conv2D/ReadVariableOp8^Representation/res_conv_A_repr_c/BiasAdd/ReadVariableOp7^Representation/res_conv_A_repr_c/Conv2D/ReadVariableOp8^Representation/res_conv_A_repr_d/BiasAdd/ReadVariableOp7^Representation/res_conv_A_repr_d/Conv2D/ReadVariableOp8^Representation/res_conv_B_repr_a/BiasAdd/ReadVariableOp7^Representation/res_conv_B_repr_a/Conv2D/ReadVariableOp8^Representation/res_conv_B_repr_b/BiasAdd/ReadVariableOp7^Representation/res_conv_B_repr_b/Conv2D/ReadVariableOp8^Representation/res_conv_B_repr_c/BiasAdd/ReadVariableOp7^Representation/res_conv_B_repr_c/Conv2D/ReadVariableOp8^Representation/res_conv_B_repr_d/BiasAdd/ReadVariableOp7^Representation/res_conv_B_repr_d/Conv2D/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������``::::::::::::::::::::::::::::::::::::::::::::::::::::::2�
BRepresentation/batch_normalization/FusedBatchNormV3/ReadVariableOpBRepresentation/batch_normalization/FusedBatchNormV3/ReadVariableOp2�
DRepresentation/batch_normalization/FusedBatchNormV3/ReadVariableOp_1DRepresentation/batch_normalization/FusedBatchNormV3/ReadVariableOp_12f
1Representation/batch_normalization/ReadVariableOp1Representation/batch_normalization/ReadVariableOp2j
3Representation/batch_normalization/ReadVariableOp_13Representation/batch_normalization/ReadVariableOp_12�
DRepresentation/batch_normalization_1/FusedBatchNormV3/ReadVariableOpDRepresentation/batch_normalization_1/FusedBatchNormV3/ReadVariableOp2�
FRepresentation/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1FRepresentation/batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12j
3Representation/batch_normalization_1/ReadVariableOp3Representation/batch_normalization_1/ReadVariableOp2n
5Representation/batch_normalization_1/ReadVariableOp_15Representation/batch_normalization_1/ReadVariableOp_12�
DRepresentation/batch_normalization_2/FusedBatchNormV3/ReadVariableOpDRepresentation/batch_normalization_2/FusedBatchNormV3/ReadVariableOp2�
FRepresentation/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1FRepresentation/batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12j
3Representation/batch_normalization_2/ReadVariableOp3Representation/batch_normalization_2/ReadVariableOp2n
5Representation/batch_normalization_2/ReadVariableOp_15Representation/batch_normalization_2/ReadVariableOp_12�
DRepresentation/batch_normalization_3/FusedBatchNormV3/ReadVariableOpDRepresentation/batch_normalization_3/FusedBatchNormV3/ReadVariableOp2�
FRepresentation/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1FRepresentation/batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12j
3Representation/batch_normalization_3/ReadVariableOp3Representation/batch_normalization_3/ReadVariableOp2n
5Representation/batch_normalization_3/ReadVariableOp_15Representation/batch_normalization_3/ReadVariableOp_12�
DRepresentation/batch_normalization_4/FusedBatchNormV3/ReadVariableOpDRepresentation/batch_normalization_4/FusedBatchNormV3/ReadVariableOp2�
FRepresentation/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1FRepresentation/batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12j
3Representation/batch_normalization_4/ReadVariableOp3Representation/batch_normalization_4/ReadVariableOp2n
5Representation/batch_normalization_4/ReadVariableOp_15Representation/batch_normalization_4/ReadVariableOp_12�
DRepresentation/batch_normalization_5/FusedBatchNormV3/ReadVariableOpDRepresentation/batch_normalization_5/FusedBatchNormV3/ReadVariableOp2�
FRepresentation/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1FRepresentation/batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12j
3Representation/batch_normalization_5/ReadVariableOp3Representation/batch_normalization_5/ReadVariableOp2n
5Representation/batch_normalization_5/ReadVariableOp_15Representation/batch_normalization_5/ReadVariableOp_12�
DRepresentation/batch_normalization_6/FusedBatchNormV3/ReadVariableOpDRepresentation/batch_normalization_6/FusedBatchNormV3/ReadVariableOp2�
FRepresentation/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1FRepresentation/batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12j
3Representation/batch_normalization_6/ReadVariableOp3Representation/batch_normalization_6/ReadVariableOp2n
5Representation/batch_normalization_6/ReadVariableOp_15Representation/batch_normalization_6/ReadVariableOp_12�
DRepresentation/batch_normalization_7/FusedBatchNormV3/ReadVariableOpDRepresentation/batch_normalization_7/FusedBatchNormV3/ReadVariableOp2�
FRepresentation/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1FRepresentation/batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12j
3Representation/batch_normalization_7/ReadVariableOp3Representation/batch_normalization_7/ReadVariableOp2n
5Representation/batch_normalization_7/ReadVariableOp_15Representation/batch_normalization_7/ReadVariableOp_12\
,Representation/conv2d/BiasAdd/ReadVariableOp,Representation/conv2d/BiasAdd/ReadVariableOp2Z
+Representation/conv2d/Conv2D/ReadVariableOp+Representation/conv2d/Conv2D/ReadVariableOp2`
.Representation/conv2d_1/BiasAdd/ReadVariableOp.Representation/conv2d_1/BiasAdd/ReadVariableOp2^
-Representation/conv2d_1/Conv2D/ReadVariableOp-Representation/conv2d_1/Conv2D/ReadVariableOp2d
0Representation/repr_board/BiasAdd/ReadVariableOp0Representation/repr_board/BiasAdd/ReadVariableOp2b
/Representation/repr_board/Conv2D/ReadVariableOp/Representation/repr_board/Conv2D/ReadVariableOp2r
7Representation/res_conv_A_repr_a/BiasAdd/ReadVariableOp7Representation/res_conv_A_repr_a/BiasAdd/ReadVariableOp2p
6Representation/res_conv_A_repr_a/Conv2D/ReadVariableOp6Representation/res_conv_A_repr_a/Conv2D/ReadVariableOp2r
7Representation/res_conv_A_repr_b/BiasAdd/ReadVariableOp7Representation/res_conv_A_repr_b/BiasAdd/ReadVariableOp2p
6Representation/res_conv_A_repr_b/Conv2D/ReadVariableOp6Representation/res_conv_A_repr_b/Conv2D/ReadVariableOp2r
7Representation/res_conv_A_repr_c/BiasAdd/ReadVariableOp7Representation/res_conv_A_repr_c/BiasAdd/ReadVariableOp2p
6Representation/res_conv_A_repr_c/Conv2D/ReadVariableOp6Representation/res_conv_A_repr_c/Conv2D/ReadVariableOp2r
7Representation/res_conv_A_repr_d/BiasAdd/ReadVariableOp7Representation/res_conv_A_repr_d/BiasAdd/ReadVariableOp2p
6Representation/res_conv_A_repr_d/Conv2D/ReadVariableOp6Representation/res_conv_A_repr_d/Conv2D/ReadVariableOp2r
7Representation/res_conv_B_repr_a/BiasAdd/ReadVariableOp7Representation/res_conv_B_repr_a/BiasAdd/ReadVariableOp2p
6Representation/res_conv_B_repr_a/Conv2D/ReadVariableOp6Representation/res_conv_B_repr_a/Conv2D/ReadVariableOp2r
7Representation/res_conv_B_repr_b/BiasAdd/ReadVariableOp7Representation/res_conv_B_repr_b/BiasAdd/ReadVariableOp2p
6Representation/res_conv_B_repr_b/Conv2D/ReadVariableOp6Representation/res_conv_B_repr_b/Conv2D/ReadVariableOp2r
7Representation/res_conv_B_repr_c/BiasAdd/ReadVariableOp7Representation/res_conv_B_repr_c/BiasAdd/ReadVariableOp2p
6Representation/res_conv_B_repr_c/Conv2D/ReadVariableOp6Representation/res_conv_B_repr_c/Conv2D/ReadVariableOp2r
7Representation/res_conv_B_repr_d/BiasAdd/ReadVariableOp7Representation/res_conv_B_repr_d/BiasAdd/ReadVariableOp2p
6Representation/res_conv_B_repr_d/Conv2D/ReadVariableOp6Representation/res_conv_B_repr_d/Conv2D/ReadVariableOp:% !

_user_specified_nameboard
�
�
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_27170

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
L__inference_res_conv_B_repr_c_layer_call_and_return_conditional_losses_22715

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAdd�
:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource^Conv2D/ReadVariableOp*&
_output_shapes
:@@*
dtype02<
:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_c/kernel/Regularizer/SquareSquareBres_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_B_repr_c/kernel/Regularizer/Square�
*res_conv_B_repr_c/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_c/kernel/Regularizer/Const�
(res_conv_B_repr_c/kernel/Regularizer/SumSum/res_conv_B_repr_c/kernel/Regularizer/Square:y:03res_conv_B_repr_c/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_c/kernel/Regularizer/Sum�
*res_conv_B_repr_c/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_c/kernel/Regularizer/mul/x�
(res_conv_B_repr_c/kernel/Regularizer/mulMul3res_conv_B_repr_c/kernel/Regularizer/mul/x:output:01res_conv_B_repr_c/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_c/kernel/Regularizer/mul�
*res_conv_B_repr_c/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_c/kernel/Regularizer/add/x�
(res_conv_B_repr_c/kernel/Regularizer/addAddV23res_conv_B_repr_c/kernel/Regularizer/add/x:output:0,res_conv_B_repr_c/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_c/kernel/Regularizer/add�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp;^res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2x
:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs
�$
�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26412

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_26397
assignmovingavg_1_26404
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/26397*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/26397*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_26397*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/26397*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/26397*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_26397AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/26397*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/26404*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/26404*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_26404*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/26404*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/26404*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_26404AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/26404*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
H
,__inference_activation_5_layer_call_fn_26912

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_237942
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_23968

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�%
�
E__inference_repr_board_layer_call_and_return_conditional_losses_23216

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�1repr_board/bias/Regularizer/Square/ReadVariableOp�3repr_board/kernel/Regularizer/Square/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+���������������������������2
Relu�
3repr_board/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource^Conv2D/ReadVariableOp*&
_output_shapes
:@*
dtype025
3repr_board/kernel/Regularizer/Square/ReadVariableOp�
$repr_board/kernel/Regularizer/SquareSquare;repr_board/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2&
$repr_board/kernel/Regularizer/Square�
#repr_board/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2%
#repr_board/kernel/Regularizer/Const�
!repr_board/kernel/Regularizer/SumSum(repr_board/kernel/Regularizer/Square:y:0,repr_board/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!repr_board/kernel/Regularizer/Sum�
#repr_board/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72%
#repr_board/kernel/Regularizer/mul/x�
!repr_board/kernel/Regularizer/mulMul,repr_board/kernel/Regularizer/mul/x:output:0*repr_board/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!repr_board/kernel/Regularizer/mul�
#repr_board/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#repr_board/kernel/Regularizer/add/x�
!repr_board/kernel/Regularizer/addAddV2,repr_board/kernel/Regularizer/add/x:output:0%repr_board/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2#
!repr_board/kernel/Regularizer/add�
1repr_board/bias/Regularizer/Square/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource^BiasAdd/ReadVariableOp*
_output_shapes
:*
dtype023
1repr_board/bias/Regularizer/Square/ReadVariableOp�
"repr_board/bias/Regularizer/SquareSquare9repr_board/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2$
"repr_board/bias/Regularizer/Square�
!repr_board/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!repr_board/bias/Regularizer/Const�
repr_board/bias/Regularizer/SumSum&repr_board/bias/Regularizer/Square:y:0*repr_board/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
repr_board/bias/Regularizer/Sum�
!repr_board/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72#
!repr_board/bias/Regularizer/mul/x�
repr_board/bias/Regularizer/mulMul*repr_board/bias/Regularizer/mul/x:output:0(repr_board/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
repr_board/bias/Regularizer/mul�
!repr_board/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!repr_board/bias/Regularizer/add/x�
repr_board/bias/Regularizer/addAddV2*repr_board/bias/Regularizer/add/x:output:0#repr_board/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
repr_board/bias/Regularizer/add�
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp2^repr_board/bias/Regularizer/Square/ReadVariableOp4^repr_board/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2f
1repr_board/bias/Regularizer/Square/ReadVariableOp1repr_board/bias/Regularizer/Square/ReadVariableOp2j
3repr_board/kernel/Regularizer/Square/ReadVariableOp3repr_board/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_22688

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
L__inference_res_conv_A_repr_c_layer_call_and_return_conditional_losses_22555

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAdd�
:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource^Conv2D/ReadVariableOp*&
_output_shapes
:@@*
dtype02<
:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_c/kernel/Regularizer/SquareSquareBres_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_A_repr_c/kernel/Regularizer/Square�
*res_conv_A_repr_c/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_c/kernel/Regularizer/Const�
(res_conv_A_repr_c/kernel/Regularizer/SumSum/res_conv_A_repr_c/kernel/Regularizer/Square:y:03res_conv_A_repr_c/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_c/kernel/Regularizer/Sum�
*res_conv_A_repr_c/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_c/kernel/Regularizer/mul/x�
(res_conv_A_repr_c/kernel/Regularizer/mulMul3res_conv_A_repr_c/kernel/Regularizer/mul/x:output:01res_conv_A_repr_c/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_c/kernel/Regularizer/mul�
*res_conv_A_repr_c/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_c/kernel/Regularizer/add/x�
(res_conv_A_repr_c/kernel/Regularizer/addAddV23res_conv_A_repr_c/kernel/Regularizer/add/x:output:0,res_conv_A_repr_c/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_c/kernel/Regularizer/add�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp;^res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2x
:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs
�
O
3__inference_average_pooling2d_1_layer_call_fn_22867

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*J
_output_shapes8
6:4������������������������������������*/
config_proto

GPU

CPU2 *0J 8*W
fRRP
N__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_228612
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_1_layer_call_fn_26253

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_234212
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������00 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������00 ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_4_layer_call_fn_26811

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_226572
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
__inference_loss_fn_2_27507G
Cres_conv_a_repr_b_kernel_regularizer_square_readvariableop_resource
identity��:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp�
:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOpReadVariableOpCres_conv_a_repr_b_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@@*
dtype02<
:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_b/kernel/Regularizer/SquareSquareBres_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_A_repr_b/kernel/Regularizer/Square�
*res_conv_A_repr_b/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_b/kernel/Regularizer/Const�
(res_conv_A_repr_b/kernel/Regularizer/SumSum/res_conv_A_repr_b/kernel/Regularizer/Square:y:03res_conv_A_repr_b/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_b/kernel/Regularizer/Sum�
*res_conv_A_repr_b/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_b/kernel/Regularizer/mul/x�
(res_conv_A_repr_b/kernel/Regularizer/mulMul3res_conv_A_repr_b/kernel/Regularizer/mul/x:output:01res_conv_A_repr_b/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_b/kernel/Regularizer/mul�
*res_conv_A_repr_b/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_b/kernel/Regularizer/add/x�
(res_conv_A_repr_b/kernel/Regularizer/addAddV23res_conv_A_repr_b/kernel/Regularizer/add/x:output:0,res_conv_A_repr_b/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_b/kernel/Regularizer/add�
IdentityIdentity,res_conv_A_repr_b/kernel/Regularizer/add:z:0;^res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2x
:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp
�$
�
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_22817

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_22802
assignmovingavg_1_22809
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/22802*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/22802*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_22802*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/22802*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/22802*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_22802AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/22802*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/22809*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/22809*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_22809*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/22809*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/22809*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_22809AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/22809*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
1__inference_res_conv_B_repr_d_layer_call_fn_23055

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_B_repr_d_layer_call_and_return_conditional_losses_230472
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�$
�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_26664

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_26649
assignmovingavg_1_26656
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/26649*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/26649*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_26649*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/26649*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/26649*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_26649AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/26649*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/26656*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/26656*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_26656*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/26656*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/26656*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_26656AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/26656*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_26980

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_6_layer_call_fn_27253

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_239462
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
H
,__inference_activation_4_layer_call_fn_26734

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_236992
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
H
,__inference_activation_6_layer_call_fn_27102

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_6_layer_call_and_return_conditional_losses_239052
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
�
1__inference_res_conv_A_repr_b_layer_call_fn_22231

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_A_repr_b_layer_call_and_return_conditional_losses_222232
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
C
'__inference_permute_layer_call_fn_21843

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*W
_output_shapesE
C:A���������������������������������������������*/
config_proto

GPU

CPU2 *0J 8*K
fFRD
B__inference_permute_layer_call_and_return_conditional_losses_218372
PartitionedCall�
IdentityIdentityPartitionedCall:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A���������������������������������������������:& "
 
_user_specified_nameinputs
�
c
G__inference_activation_5_layer_call_and_return_conditional_losses_23794

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
c
G__inference_activation_4_layer_call_and_return_conditional_losses_23699

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�%
�
.__inference_Representation_layer_call_fn_24827	
board"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36#
statefulpartitionedcall_args_37#
statefulpartitionedcall_args_38#
statefulpartitionedcall_args_39#
statefulpartitionedcall_args_40#
statefulpartitionedcall_args_41#
statefulpartitionedcall_args_42#
statefulpartitionedcall_args_43#
statefulpartitionedcall_args_44#
statefulpartitionedcall_args_45#
statefulpartitionedcall_args_46#
statefulpartitionedcall_args_47#
statefulpartitionedcall_args_48#
statefulpartitionedcall_args_49#
statefulpartitionedcall_args_50#
statefulpartitionedcall_args_51#
statefulpartitionedcall_args_52#
statefulpartitionedcall_args_53#
statefulpartitionedcall_args_54
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallboardstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36statefulpartitionedcall_args_37statefulpartitionedcall_args_38statefulpartitionedcall_args_39statefulpartitionedcall_args_40statefulpartitionedcall_args_41statefulpartitionedcall_args_42statefulpartitionedcall_args_43statefulpartitionedcall_args_44statefulpartitionedcall_args_45statefulpartitionedcall_args_46statefulpartitionedcall_args_47statefulpartitionedcall_args_48statefulpartitionedcall_args_49statefulpartitionedcall_args_50statefulpartitionedcall_args_51statefulpartitionedcall_args_52statefulpartitionedcall_args_53statefulpartitionedcall_args_54*B
Tin;
927*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������*/
config_proto

GPU

CPU2 *0J 8*R
fMRK
I__inference_Representation_layer_call_and_return_conditional_losses_247702
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������``::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:% !

_user_specified_nameboard
�
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_25975

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :`2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :`2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������``2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������``2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������``:& "
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_26318

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�#
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_23326

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_23311
assignmovingavg_1_23318
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������00 : : : : :*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/23311*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/23311*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23311*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/23311*
_output_shapes
: 2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/23311*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23311AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/23311*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/23318*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/23318*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23318*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/23318*
_output_shapes
: 2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/23318*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23318AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/23318*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������00 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������00 ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�%
�
#__inference_signature_wrapper_25063	
board"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36#
statefulpartitionedcall_args_37#
statefulpartitionedcall_args_38#
statefulpartitionedcall_args_39#
statefulpartitionedcall_args_40#
statefulpartitionedcall_args_41#
statefulpartitionedcall_args_42#
statefulpartitionedcall_args_43#
statefulpartitionedcall_args_44#
statefulpartitionedcall_args_45#
statefulpartitionedcall_args_46#
statefulpartitionedcall_args_47#
statefulpartitionedcall_args_48#
statefulpartitionedcall_args_49#
statefulpartitionedcall_args_50#
statefulpartitionedcall_args_51#
statefulpartitionedcall_args_52#
statefulpartitionedcall_args_53#
statefulpartitionedcall_args_54
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallboardstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36statefulpartitionedcall_args_37statefulpartitionedcall_args_38statefulpartitionedcall_args_39statefulpartitionedcall_args_40statefulpartitionedcall_args_41statefulpartitionedcall_args_42statefulpartitionedcall_args_43statefulpartitionedcall_args_44statefulpartitionedcall_args_45statefulpartitionedcall_args_46statefulpartitionedcall_args_47statefulpartitionedcall_args_48statefulpartitionedcall_args_49statefulpartitionedcall_args_50statefulpartitionedcall_args_51statefulpartitionedcall_args_52statefulpartitionedcall_args_53statefulpartitionedcall_args_54*B
Tin;
927*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������*/
config_proto

GPU

CPU2 *0J 8*)
f$R"
 __inference__wrapped_model_218302
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������``::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:% !

_user_specified_nameboard
�
�
__inference_loss_fn_0_27481G
Cres_conv_a_repr_a_kernel_regularizer_square_readvariableop_resource
identity��:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp�
:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOpReadVariableOpCres_conv_a_repr_a_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:  *
dtype02<
:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_a/kernel/Regularizer/SquareSquareBres_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2-
+res_conv_A_repr_a/kernel/Regularizer/Square�
*res_conv_A_repr_a/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_a/kernel/Regularizer/Const�
(res_conv_A_repr_a/kernel/Regularizer/SumSum/res_conv_A_repr_a/kernel/Regularizer/Square:y:03res_conv_A_repr_a/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_a/kernel/Regularizer/Sum�
*res_conv_A_repr_a/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_a/kernel/Regularizer/mul/x�
(res_conv_A_repr_a/kernel/Regularizer/mulMul3res_conv_A_repr_a/kernel/Regularizer/mul/x:output:01res_conv_A_repr_a/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_a/kernel/Regularizer/mul�
*res_conv_A_repr_a/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_a/kernel/Regularizer/add/x�
(res_conv_A_repr_a/kernel/Regularizer/addAddV23res_conv_A_repr_a/kernel/Regularizer/add/x:output:0,res_conv_A_repr_a/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_a/kernel/Regularizer/add�
IdentityIdentity,res_conv_A_repr_a/kernel/Regularizer/add:z:0;^res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2x
:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp
��
�5
I__inference_Representation_layer_call_and_return_conditional_losses_25491

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource4
0res_conv_a_repr_a_conv2d_readvariableop_resource5
1res_conv_a_repr_a_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource-
)batch_normalization_assignmovingavg_25118/
+batch_normalization_assignmovingavg_1_251254
0res_conv_b_repr_a_conv2d_readvariableop_resource5
1res_conv_b_repr_a_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resource/
+batch_normalization_1_assignmovingavg_251551
-batch_normalization_1_assignmovingavg_1_25162+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource4
0res_conv_a_repr_b_conv2d_readvariableop_resource5
1res_conv_a_repr_b_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resource/
+batch_normalization_2_assignmovingavg_251991
-batch_normalization_2_assignmovingavg_1_252064
0res_conv_b_repr_b_conv2d_readvariableop_resource5
1res_conv_b_repr_b_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resource/
+batch_normalization_3_assignmovingavg_252361
-batch_normalization_3_assignmovingavg_1_252434
0res_conv_a_repr_c_conv2d_readvariableop_resource5
1res_conv_a_repr_c_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resource/
+batch_normalization_4_assignmovingavg_252751
-batch_normalization_4_assignmovingavg_1_252824
0res_conv_b_repr_c_conv2d_readvariableop_resource5
1res_conv_b_repr_c_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resource/
+batch_normalization_5_assignmovingavg_253121
-batch_normalization_5_assignmovingavg_1_253194
0res_conv_a_repr_d_conv2d_readvariableop_resource5
1res_conv_a_repr_d_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resource/
+batch_normalization_6_assignmovingavg_253511
-batch_normalization_6_assignmovingavg_1_253584
0res_conv_b_repr_d_conv2d_readvariableop_resource5
1res_conv_b_repr_d_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resource/
+batch_normalization_7_assignmovingavg_253881
-batch_normalization_7_assignmovingavg_1_25395-
)repr_board_conv2d_readvariableop_resource.
*repr_board_biasadd_readvariableop_resource
identity��7batch_normalization/AssignMovingAvg/AssignSubVariableOp�2batch_normalization/AssignMovingAvg/ReadVariableOp�9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp�4batch_normalization/AssignMovingAvg_1/ReadVariableOp�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_1/AssignMovingAvg/ReadVariableOp�;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�$batch_normalization_1/ReadVariableOp�&batch_normalization_1/ReadVariableOp_1�9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_2/AssignMovingAvg/ReadVariableOp�;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�$batch_normalization_2/ReadVariableOp�&batch_normalization_2/ReadVariableOp_1�9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_3/AssignMovingAvg/ReadVariableOp�;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp�$batch_normalization_3/ReadVariableOp�&batch_normalization_3/ReadVariableOp_1�9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_4/AssignMovingAvg/ReadVariableOp�;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp�$batch_normalization_4/ReadVariableOp�&batch_normalization_4/ReadVariableOp_1�9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_5/AssignMovingAvg/ReadVariableOp�;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp�$batch_normalization_5/ReadVariableOp�&batch_normalization_5/ReadVariableOp_1�9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_6/AssignMovingAvg/ReadVariableOp�;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp�$batch_normalization_6/ReadVariableOp�&batch_normalization_6/ReadVariableOp_1�9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp�4batch_normalization_7/AssignMovingAvg/ReadVariableOp�;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp�6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp�$batch_normalization_7/ReadVariableOp�&batch_normalization_7/ReadVariableOp_1�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�!repr_board/BiasAdd/ReadVariableOp� repr_board/Conv2D/ReadVariableOp�1repr_board/bias/Regularizer/Square/ReadVariableOp�3repr_board/kernel/Regularizer/Square/ReadVariableOp�(res_conv_A_repr_a/BiasAdd/ReadVariableOp�'res_conv_A_repr_a/Conv2D/ReadVariableOp�:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp�(res_conv_A_repr_b/BiasAdd/ReadVariableOp�'res_conv_A_repr_b/Conv2D/ReadVariableOp�:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp�(res_conv_A_repr_c/BiasAdd/ReadVariableOp�'res_conv_A_repr_c/Conv2D/ReadVariableOp�:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp�(res_conv_A_repr_d/BiasAdd/ReadVariableOp�'res_conv_A_repr_d/Conv2D/ReadVariableOp�:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp�(res_conv_B_repr_a/BiasAdd/ReadVariableOp�'res_conv_B_repr_a/Conv2D/ReadVariableOp�:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp�(res_conv_B_repr_b/BiasAdd/ReadVariableOp�'res_conv_B_repr_b/Conv2D/ReadVariableOp�:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp�(res_conv_B_repr_c/BiasAdd/ReadVariableOp�'res_conv_B_repr_c/Conv2D/ReadVariableOp�:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp�(res_conv_B_repr_d/BiasAdd/ReadVariableOp�'res_conv_B_repr_d/Conv2D/ReadVariableOp�:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOpT
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/Shape�
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack�
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1�
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :`2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :`2
reshape/Reshape/shape/3t
reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/4�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0 reshape/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape�
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*3
_output_shapes!
:���������``2
reshape/Reshape�
permute/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
permute/transpose/perm�
permute/transpose	Transposereshape/Reshape:output:0permute/transpose/perm:output:0*
T0*3
_output_shapes!
:���������``2
permute/transposeg
reshape_1/ShapeShapepermute/transpose:y:0*
T0*
_output_shapes
:2
reshape_1/Shape�
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack�
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1�
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2�
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :`2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :`2
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/3�
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape�
reshape_1/ReshapeReshapepermute/transpose:y:0 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������``2
reshape_1/Reshape�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp�
conv2d/Conv2DConv2Dreshape_1/Reshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00 *
paddingSAME*
strides
2
conv2d/Conv2D�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00 2
conv2d/BiasAdd�
'res_conv_A_repr_a/Conv2D/ReadVariableOpReadVariableOp0res_conv_a_repr_a_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'res_conv_A_repr_a/Conv2D/ReadVariableOp�
res_conv_A_repr_a/Conv2DConv2Dconv2d/BiasAdd:output:0/res_conv_A_repr_a/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00 *
paddingSAME*
strides
2
res_conv_A_repr_a/Conv2D�
(res_conv_A_repr_a/BiasAdd/ReadVariableOpReadVariableOp1res_conv_a_repr_a_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(res_conv_A_repr_a/BiasAdd/ReadVariableOp�
res_conv_A_repr_a/BiasAddBiasAdd!res_conv_A_repr_a/Conv2D:output:00res_conv_A_repr_a/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00 2
res_conv_A_repr_a/BiasAdd�
activation/ReluRelu"res_conv_A_repr_a/BiasAdd:output:0*
T0*/
_output_shapes
:���������00 2
activation/Relu�
 batch_normalization/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2"
 batch_normalization/LogicalAnd/x�
 batch_normalization/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2"
 batch_normalization/LogicalAnd/y�
batch_normalization/LogicalAnd
LogicalAnd)batch_normalization/LogicalAnd/x:output:0)batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: 2 
batch_normalization/LogicalAnd�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1y
batch_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization/Const}
batch_normalization/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization/Const_1�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3activation/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0"batch_normalization/Const:output:0$batch_normalization/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������00 : : : : :*
epsilon%o�:2&
$batch_normalization/FusedBatchNormV3
batch_normalization/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization/Const_2�
)batch_normalization/AssignMovingAvg/sub/xConst*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/25118*
_output_shapes
: *
dtype0*
valueB
 *  �?2+
)batch_normalization/AssignMovingAvg/sub/x�
'batch_normalization/AssignMovingAvg/subSub2batch_normalization/AssignMovingAvg/sub/x:output:0$batch_normalization/Const_2:output:0*
T0*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/25118*
_output_shapes
: 2)
'batch_normalization/AssignMovingAvg/sub�
2batch_normalization/AssignMovingAvg/ReadVariableOpReadVariableOp)batch_normalization_assignmovingavg_25118*
_output_shapes
: *
dtype024
2batch_normalization/AssignMovingAvg/ReadVariableOp�
)batch_normalization/AssignMovingAvg/sub_1Sub:batch_normalization/AssignMovingAvg/ReadVariableOp:value:01batch_normalization/FusedBatchNormV3:batch_mean:0*
T0*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/25118*
_output_shapes
: 2+
)batch_normalization/AssignMovingAvg/sub_1�
'batch_normalization/AssignMovingAvg/mulMul-batch_normalization/AssignMovingAvg/sub_1:z:0+batch_normalization/AssignMovingAvg/sub:z:0*
T0*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/25118*
_output_shapes
: 2)
'batch_normalization/AssignMovingAvg/mul�
7batch_normalization/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp)batch_normalization_assignmovingavg_25118+batch_normalization/AssignMovingAvg/mul:z:03^batch_normalization/AssignMovingAvg/ReadVariableOp*<
_class2
0.loc:@batch_normalization/AssignMovingAvg/25118*
_output_shapes
 *
dtype029
7batch_normalization/AssignMovingAvg/AssignSubVariableOp�
+batch_normalization/AssignMovingAvg_1/sub/xConst*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/25125*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+batch_normalization/AssignMovingAvg_1/sub/x�
)batch_normalization/AssignMovingAvg_1/subSub4batch_normalization/AssignMovingAvg_1/sub/x:output:0$batch_normalization/Const_2:output:0*
T0*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/25125*
_output_shapes
: 2+
)batch_normalization/AssignMovingAvg_1/sub�
4batch_normalization/AssignMovingAvg_1/ReadVariableOpReadVariableOp+batch_normalization_assignmovingavg_1_25125*
_output_shapes
: *
dtype026
4batch_normalization/AssignMovingAvg_1/ReadVariableOp�
+batch_normalization/AssignMovingAvg_1/sub_1Sub<batch_normalization/AssignMovingAvg_1/ReadVariableOp:value:05batch_normalization/FusedBatchNormV3:batch_variance:0*
T0*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/25125*
_output_shapes
: 2-
+batch_normalization/AssignMovingAvg_1/sub_1�
)batch_normalization/AssignMovingAvg_1/mulMul/batch_normalization/AssignMovingAvg_1/sub_1:z:0-batch_normalization/AssignMovingAvg_1/sub:z:0*
T0*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/25125*
_output_shapes
: 2+
)batch_normalization/AssignMovingAvg_1/mul�
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp+batch_normalization_assignmovingavg_1_25125-batch_normalization/AssignMovingAvg_1/mul:z:05^batch_normalization/AssignMovingAvg_1/ReadVariableOp*>
_class4
20loc:@batch_normalization/AssignMovingAvg_1/25125*
_output_shapes
 *
dtype02;
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp�
'res_conv_B_repr_a/Conv2D/ReadVariableOpReadVariableOp0res_conv_b_repr_a_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'res_conv_B_repr_a/Conv2D/ReadVariableOp�
res_conv_B_repr_a/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0/res_conv_B_repr_a/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00 *
paddingSAME*
strides
2
res_conv_B_repr_a/Conv2D�
(res_conv_B_repr_a/BiasAdd/ReadVariableOpReadVariableOp1res_conv_b_repr_a_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(res_conv_B_repr_a/BiasAdd/ReadVariableOp�
res_conv_B_repr_a/BiasAddBiasAdd!res_conv_B_repr_a/Conv2D:output:00res_conv_B_repr_a/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00 2
res_conv_B_repr_a/BiasAdd�
activation_1/ReluRelu"res_conv_B_repr_a/BiasAdd:output:0*
T0*/
_output_shapes
:���������00 2
activation_1/Relu�
"batch_normalization_1/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_1/LogicalAnd/x�
"batch_normalization_1/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_1/LogicalAnd/y�
 batch_normalization_1/LogicalAnd
LogicalAnd+batch_normalization_1/LogicalAnd/x:output:0+batch_normalization_1/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_1/LogicalAnd�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1}
batch_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_1/Const�
batch_normalization_1/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_1/Const_1�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3activation_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0$batch_normalization_1/Const:output:0&batch_normalization_1/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������00 : : : : :*
epsilon%o�:2(
&batch_normalization_1/FusedBatchNormV3�
batch_normalization_1/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_1/Const_2�
+batch_normalization_1/AssignMovingAvg/sub/xConst*>
_class4
20loc:@batch_normalization_1/AssignMovingAvg/25155*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+batch_normalization_1/AssignMovingAvg/sub/x�
)batch_normalization_1/AssignMovingAvg/subSub4batch_normalization_1/AssignMovingAvg/sub/x:output:0&batch_normalization_1/Const_2:output:0*
T0*>
_class4
20loc:@batch_normalization_1/AssignMovingAvg/25155*
_output_shapes
: 2+
)batch_normalization_1/AssignMovingAvg/sub�
4batch_normalization_1/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_1_assignmovingavg_25155*
_output_shapes
: *
dtype026
4batch_normalization_1/AssignMovingAvg/ReadVariableOp�
+batch_normalization_1/AssignMovingAvg/sub_1Sub<batch_normalization_1/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_1/FusedBatchNormV3:batch_mean:0*
T0*>
_class4
20loc:@batch_normalization_1/AssignMovingAvg/25155*
_output_shapes
: 2-
+batch_normalization_1/AssignMovingAvg/sub_1�
)batch_normalization_1/AssignMovingAvg/mulMul/batch_normalization_1/AssignMovingAvg/sub_1:z:0-batch_normalization_1/AssignMovingAvg/sub:z:0*
T0*>
_class4
20loc:@batch_normalization_1/AssignMovingAvg/25155*
_output_shapes
: 2+
)batch_normalization_1/AssignMovingAvg/mul�
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_1_assignmovingavg_25155-batch_normalization_1/AssignMovingAvg/mul:z:05^batch_normalization_1/AssignMovingAvg/ReadVariableOp*>
_class4
20loc:@batch_normalization_1/AssignMovingAvg/25155*
_output_shapes
 *
dtype02;
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_1/AssignMovingAvg_1/sub/xConst*@
_class6
42loc:@batch_normalization_1/AssignMovingAvg_1/25162*
_output_shapes
: *
dtype0*
valueB
 *  �?2/
-batch_normalization_1/AssignMovingAvg_1/sub/x�
+batch_normalization_1/AssignMovingAvg_1/subSub6batch_normalization_1/AssignMovingAvg_1/sub/x:output:0&batch_normalization_1/Const_2:output:0*
T0*@
_class6
42loc:@batch_normalization_1/AssignMovingAvg_1/25162*
_output_shapes
: 2-
+batch_normalization_1/AssignMovingAvg_1/sub�
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_1_assignmovingavg_1_25162*
_output_shapes
: *
dtype028
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp�
-batch_normalization_1/AssignMovingAvg_1/sub_1Sub>batch_normalization_1/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_1/FusedBatchNormV3:batch_variance:0*
T0*@
_class6
42loc:@batch_normalization_1/AssignMovingAvg_1/25162*
_output_shapes
: 2/
-batch_normalization_1/AssignMovingAvg_1/sub_1�
+batch_normalization_1/AssignMovingAvg_1/mulMul1batch_normalization_1/AssignMovingAvg_1/sub_1:z:0/batch_normalization_1/AssignMovingAvg_1/sub:z:0*
T0*@
_class6
42loc:@batch_normalization_1/AssignMovingAvg_1/25162*
_output_shapes
: 2-
+batch_normalization_1/AssignMovingAvg_1/mul�
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_1_assignmovingavg_1_25162/batch_normalization_1/AssignMovingAvg_1/mul:z:07^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp*@
_class6
42loc:@batch_normalization_1/AssignMovingAvg_1/25162*
_output_shapes
 *
dtype02=
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp�
add/addAddV2*batch_normalization_1/FusedBatchNormV3:y:0conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������00 2	
add/add�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp�
conv2d_1/Conv2DConv2Dadd/add:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_1/Conv2D�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_1/BiasAdd�
'res_conv_A_repr_b/Conv2D/ReadVariableOpReadVariableOp0res_conv_a_repr_b_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'res_conv_A_repr_b/Conv2D/ReadVariableOp�
res_conv_A_repr_b/Conv2DConv2Dconv2d_1/BiasAdd:output:0/res_conv_A_repr_b/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
res_conv_A_repr_b/Conv2D�
(res_conv_A_repr_b/BiasAdd/ReadVariableOpReadVariableOp1res_conv_a_repr_b_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(res_conv_A_repr_b/BiasAdd/ReadVariableOp�
res_conv_A_repr_b/BiasAddBiasAdd!res_conv_A_repr_b/Conv2D:output:00res_conv_A_repr_b/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
res_conv_A_repr_b/BiasAdd�
activation_2/ReluRelu"res_conv_A_repr_b/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
activation_2/Relu�
"batch_normalization_2/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_2/LogicalAnd/x�
"batch_normalization_2/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_2/LogicalAnd/y�
 batch_normalization_2/LogicalAnd
LogicalAnd+batch_normalization_2/LogicalAnd/x:output:0+batch_normalization_2/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_2/LogicalAnd�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1}
batch_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_2/Const�
batch_normalization_2/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_2/Const_1�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3activation_2/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0$batch_normalization_2/Const:output:0&batch_normalization_2/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2(
&batch_normalization_2/FusedBatchNormV3�
batch_normalization_2/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_2/Const_2�
+batch_normalization_2/AssignMovingAvg/sub/xConst*>
_class4
20loc:@batch_normalization_2/AssignMovingAvg/25199*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+batch_normalization_2/AssignMovingAvg/sub/x�
)batch_normalization_2/AssignMovingAvg/subSub4batch_normalization_2/AssignMovingAvg/sub/x:output:0&batch_normalization_2/Const_2:output:0*
T0*>
_class4
20loc:@batch_normalization_2/AssignMovingAvg/25199*
_output_shapes
: 2+
)batch_normalization_2/AssignMovingAvg/sub�
4batch_normalization_2/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_2_assignmovingavg_25199*
_output_shapes
:@*
dtype026
4batch_normalization_2/AssignMovingAvg/ReadVariableOp�
+batch_normalization_2/AssignMovingAvg/sub_1Sub<batch_normalization_2/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_2/FusedBatchNormV3:batch_mean:0*
T0*>
_class4
20loc:@batch_normalization_2/AssignMovingAvg/25199*
_output_shapes
:@2-
+batch_normalization_2/AssignMovingAvg/sub_1�
)batch_normalization_2/AssignMovingAvg/mulMul/batch_normalization_2/AssignMovingAvg/sub_1:z:0-batch_normalization_2/AssignMovingAvg/sub:z:0*
T0*>
_class4
20loc:@batch_normalization_2/AssignMovingAvg/25199*
_output_shapes
:@2+
)batch_normalization_2/AssignMovingAvg/mul�
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_2_assignmovingavg_25199-batch_normalization_2/AssignMovingAvg/mul:z:05^batch_normalization_2/AssignMovingAvg/ReadVariableOp*>
_class4
20loc:@batch_normalization_2/AssignMovingAvg/25199*
_output_shapes
 *
dtype02;
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_2/AssignMovingAvg_1/sub/xConst*@
_class6
42loc:@batch_normalization_2/AssignMovingAvg_1/25206*
_output_shapes
: *
dtype0*
valueB
 *  �?2/
-batch_normalization_2/AssignMovingAvg_1/sub/x�
+batch_normalization_2/AssignMovingAvg_1/subSub6batch_normalization_2/AssignMovingAvg_1/sub/x:output:0&batch_normalization_2/Const_2:output:0*
T0*@
_class6
42loc:@batch_normalization_2/AssignMovingAvg_1/25206*
_output_shapes
: 2-
+batch_normalization_2/AssignMovingAvg_1/sub�
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_2_assignmovingavg_1_25206*
_output_shapes
:@*
dtype028
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp�
-batch_normalization_2/AssignMovingAvg_1/sub_1Sub>batch_normalization_2/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_2/FusedBatchNormV3:batch_variance:0*
T0*@
_class6
42loc:@batch_normalization_2/AssignMovingAvg_1/25206*
_output_shapes
:@2/
-batch_normalization_2/AssignMovingAvg_1/sub_1�
+batch_normalization_2/AssignMovingAvg_1/mulMul1batch_normalization_2/AssignMovingAvg_1/sub_1:z:0/batch_normalization_2/AssignMovingAvg_1/sub:z:0*
T0*@
_class6
42loc:@batch_normalization_2/AssignMovingAvg_1/25206*
_output_shapes
:@2-
+batch_normalization_2/AssignMovingAvg_1/mul�
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_2_assignmovingavg_1_25206/batch_normalization_2/AssignMovingAvg_1/mul:z:07^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp*@
_class6
42loc:@batch_normalization_2/AssignMovingAvg_1/25206*
_output_shapes
 *
dtype02=
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp�
'res_conv_B_repr_b/Conv2D/ReadVariableOpReadVariableOp0res_conv_b_repr_b_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'res_conv_B_repr_b/Conv2D/ReadVariableOp�
res_conv_B_repr_b/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0/res_conv_B_repr_b/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
res_conv_B_repr_b/Conv2D�
(res_conv_B_repr_b/BiasAdd/ReadVariableOpReadVariableOp1res_conv_b_repr_b_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(res_conv_B_repr_b/BiasAdd/ReadVariableOp�
res_conv_B_repr_b/BiasAddBiasAdd!res_conv_B_repr_b/Conv2D:output:00res_conv_B_repr_b/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
res_conv_B_repr_b/BiasAdd�
activation_3/ReluRelu"res_conv_B_repr_b/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
activation_3/Relu�
"batch_normalization_3/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_3/LogicalAnd/x�
"batch_normalization_3/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_3/LogicalAnd/y�
 batch_normalization_3/LogicalAnd
LogicalAnd+batch_normalization_3/LogicalAnd/x:output:0+batch_normalization_3/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_3/LogicalAnd�
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOp�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1}
batch_normalization_3/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_3/Const�
batch_normalization_3/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_3/Const_1�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3activation_3/Relu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0$batch_normalization_3/Const:output:0&batch_normalization_3/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2(
&batch_normalization_3/FusedBatchNormV3�
batch_normalization_3/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_3/Const_2�
+batch_normalization_3/AssignMovingAvg/sub/xConst*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/25236*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+batch_normalization_3/AssignMovingAvg/sub/x�
)batch_normalization_3/AssignMovingAvg/subSub4batch_normalization_3/AssignMovingAvg/sub/x:output:0&batch_normalization_3/Const_2:output:0*
T0*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/25236*
_output_shapes
: 2+
)batch_normalization_3/AssignMovingAvg/sub�
4batch_normalization_3/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_3_assignmovingavg_25236*
_output_shapes
:@*
dtype026
4batch_normalization_3/AssignMovingAvg/ReadVariableOp�
+batch_normalization_3/AssignMovingAvg/sub_1Sub<batch_normalization_3/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_3/FusedBatchNormV3:batch_mean:0*
T0*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/25236*
_output_shapes
:@2-
+batch_normalization_3/AssignMovingAvg/sub_1�
)batch_normalization_3/AssignMovingAvg/mulMul/batch_normalization_3/AssignMovingAvg/sub_1:z:0-batch_normalization_3/AssignMovingAvg/sub:z:0*
T0*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/25236*
_output_shapes
:@2+
)batch_normalization_3/AssignMovingAvg/mul�
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_3_assignmovingavg_25236-batch_normalization_3/AssignMovingAvg/mul:z:05^batch_normalization_3/AssignMovingAvg/ReadVariableOp*>
_class4
20loc:@batch_normalization_3/AssignMovingAvg/25236*
_output_shapes
 *
dtype02;
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_3/AssignMovingAvg_1/sub/xConst*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/25243*
_output_shapes
: *
dtype0*
valueB
 *  �?2/
-batch_normalization_3/AssignMovingAvg_1/sub/x�
+batch_normalization_3/AssignMovingAvg_1/subSub6batch_normalization_3/AssignMovingAvg_1/sub/x:output:0&batch_normalization_3/Const_2:output:0*
T0*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/25243*
_output_shapes
: 2-
+batch_normalization_3/AssignMovingAvg_1/sub�
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_3_assignmovingavg_1_25243*
_output_shapes
:@*
dtype028
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp�
-batch_normalization_3/AssignMovingAvg_1/sub_1Sub>batch_normalization_3/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_3/FusedBatchNormV3:batch_variance:0*
T0*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/25243*
_output_shapes
:@2/
-batch_normalization_3/AssignMovingAvg_1/sub_1�
+batch_normalization_3/AssignMovingAvg_1/mulMul1batch_normalization_3/AssignMovingAvg_1/sub_1:z:0/batch_normalization_3/AssignMovingAvg_1/sub:z:0*
T0*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/25243*
_output_shapes
:@2-
+batch_normalization_3/AssignMovingAvg_1/mul�
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_3_assignmovingavg_1_25243/batch_normalization_3/AssignMovingAvg_1/mul:z:07^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp*@
_class6
42loc:@batch_normalization_3/AssignMovingAvg_1/25243*
_output_shapes
 *
dtype02=
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp�
	add_1/addAddV2*batch_normalization_3/FusedBatchNormV3:y:0conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
	add_1/add�
average_pooling2d/AvgPoolAvgPooladd_1/add:z:0*
T0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPool�
'res_conv_A_repr_c/Conv2D/ReadVariableOpReadVariableOp0res_conv_a_repr_c_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'res_conv_A_repr_c/Conv2D/ReadVariableOp�
res_conv_A_repr_c/Conv2DConv2D"average_pooling2d/AvgPool:output:0/res_conv_A_repr_c/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
res_conv_A_repr_c/Conv2D�
(res_conv_A_repr_c/BiasAdd/ReadVariableOpReadVariableOp1res_conv_a_repr_c_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(res_conv_A_repr_c/BiasAdd/ReadVariableOp�
res_conv_A_repr_c/BiasAddBiasAdd!res_conv_A_repr_c/Conv2D:output:00res_conv_A_repr_c/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
res_conv_A_repr_c/BiasAdd�
activation_4/ReluRelu"res_conv_A_repr_c/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
activation_4/Relu�
"batch_normalization_4/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_4/LogicalAnd/x�
"batch_normalization_4/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_4/LogicalAnd/y�
 batch_normalization_4/LogicalAnd
LogicalAnd+batch_normalization_4/LogicalAnd/x:output:0+batch_normalization_4/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_4/LogicalAnd�
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOp�
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1}
batch_normalization_4/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_4/Const�
batch_normalization_4/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_4/Const_1�
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3activation_4/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0$batch_normalization_4/Const:output:0&batch_normalization_4/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2(
&batch_normalization_4/FusedBatchNormV3�
batch_normalization_4/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_4/Const_2�
+batch_normalization_4/AssignMovingAvg/sub/xConst*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/25275*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+batch_normalization_4/AssignMovingAvg/sub/x�
)batch_normalization_4/AssignMovingAvg/subSub4batch_normalization_4/AssignMovingAvg/sub/x:output:0&batch_normalization_4/Const_2:output:0*
T0*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/25275*
_output_shapes
: 2+
)batch_normalization_4/AssignMovingAvg/sub�
4batch_normalization_4/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_4_assignmovingavg_25275*
_output_shapes
:@*
dtype026
4batch_normalization_4/AssignMovingAvg/ReadVariableOp�
+batch_normalization_4/AssignMovingAvg/sub_1Sub<batch_normalization_4/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_4/FusedBatchNormV3:batch_mean:0*
T0*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/25275*
_output_shapes
:@2-
+batch_normalization_4/AssignMovingAvg/sub_1�
)batch_normalization_4/AssignMovingAvg/mulMul/batch_normalization_4/AssignMovingAvg/sub_1:z:0-batch_normalization_4/AssignMovingAvg/sub:z:0*
T0*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/25275*
_output_shapes
:@2+
)batch_normalization_4/AssignMovingAvg/mul�
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_4_assignmovingavg_25275-batch_normalization_4/AssignMovingAvg/mul:z:05^batch_normalization_4/AssignMovingAvg/ReadVariableOp*>
_class4
20loc:@batch_normalization_4/AssignMovingAvg/25275*
_output_shapes
 *
dtype02;
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_4/AssignMovingAvg_1/sub/xConst*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/25282*
_output_shapes
: *
dtype0*
valueB
 *  �?2/
-batch_normalization_4/AssignMovingAvg_1/sub/x�
+batch_normalization_4/AssignMovingAvg_1/subSub6batch_normalization_4/AssignMovingAvg_1/sub/x:output:0&batch_normalization_4/Const_2:output:0*
T0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/25282*
_output_shapes
: 2-
+batch_normalization_4/AssignMovingAvg_1/sub�
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_4_assignmovingavg_1_25282*
_output_shapes
:@*
dtype028
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp�
-batch_normalization_4/AssignMovingAvg_1/sub_1Sub>batch_normalization_4/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_4/FusedBatchNormV3:batch_variance:0*
T0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/25282*
_output_shapes
:@2/
-batch_normalization_4/AssignMovingAvg_1/sub_1�
+batch_normalization_4/AssignMovingAvg_1/mulMul1batch_normalization_4/AssignMovingAvg_1/sub_1:z:0/batch_normalization_4/AssignMovingAvg_1/sub:z:0*
T0*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/25282*
_output_shapes
:@2-
+batch_normalization_4/AssignMovingAvg_1/mul�
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_4_assignmovingavg_1_25282/batch_normalization_4/AssignMovingAvg_1/mul:z:07^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp*@
_class6
42loc:@batch_normalization_4/AssignMovingAvg_1/25282*
_output_shapes
 *
dtype02=
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp�
'res_conv_B_repr_c/Conv2D/ReadVariableOpReadVariableOp0res_conv_b_repr_c_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'res_conv_B_repr_c/Conv2D/ReadVariableOp�
res_conv_B_repr_c/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0/res_conv_B_repr_c/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
res_conv_B_repr_c/Conv2D�
(res_conv_B_repr_c/BiasAdd/ReadVariableOpReadVariableOp1res_conv_b_repr_c_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(res_conv_B_repr_c/BiasAdd/ReadVariableOp�
res_conv_B_repr_c/BiasAddBiasAdd!res_conv_B_repr_c/Conv2D:output:00res_conv_B_repr_c/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
res_conv_B_repr_c/BiasAdd�
activation_5/ReluRelu"res_conv_B_repr_c/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
activation_5/Relu�
"batch_normalization_5/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_5/LogicalAnd/x�
"batch_normalization_5/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_5/LogicalAnd/y�
 batch_normalization_5/LogicalAnd
LogicalAnd+batch_normalization_5/LogicalAnd/x:output:0+batch_normalization_5/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_5/LogicalAnd�
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOp�
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1}
batch_normalization_5/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_5/Const�
batch_normalization_5/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_5/Const_1�
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3activation_5/Relu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0$batch_normalization_5/Const:output:0&batch_normalization_5/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2(
&batch_normalization_5/FusedBatchNormV3�
batch_normalization_5/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_5/Const_2�
+batch_normalization_5/AssignMovingAvg/sub/xConst*>
_class4
20loc:@batch_normalization_5/AssignMovingAvg/25312*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+batch_normalization_5/AssignMovingAvg/sub/x�
)batch_normalization_5/AssignMovingAvg/subSub4batch_normalization_5/AssignMovingAvg/sub/x:output:0&batch_normalization_5/Const_2:output:0*
T0*>
_class4
20loc:@batch_normalization_5/AssignMovingAvg/25312*
_output_shapes
: 2+
)batch_normalization_5/AssignMovingAvg/sub�
4batch_normalization_5/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_5_assignmovingavg_25312*
_output_shapes
:@*
dtype026
4batch_normalization_5/AssignMovingAvg/ReadVariableOp�
+batch_normalization_5/AssignMovingAvg/sub_1Sub<batch_normalization_5/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_5/FusedBatchNormV3:batch_mean:0*
T0*>
_class4
20loc:@batch_normalization_5/AssignMovingAvg/25312*
_output_shapes
:@2-
+batch_normalization_5/AssignMovingAvg/sub_1�
)batch_normalization_5/AssignMovingAvg/mulMul/batch_normalization_5/AssignMovingAvg/sub_1:z:0-batch_normalization_5/AssignMovingAvg/sub:z:0*
T0*>
_class4
20loc:@batch_normalization_5/AssignMovingAvg/25312*
_output_shapes
:@2+
)batch_normalization_5/AssignMovingAvg/mul�
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_5_assignmovingavg_25312-batch_normalization_5/AssignMovingAvg/mul:z:05^batch_normalization_5/AssignMovingAvg/ReadVariableOp*>
_class4
20loc:@batch_normalization_5/AssignMovingAvg/25312*
_output_shapes
 *
dtype02;
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_5/AssignMovingAvg_1/sub/xConst*@
_class6
42loc:@batch_normalization_5/AssignMovingAvg_1/25319*
_output_shapes
: *
dtype0*
valueB
 *  �?2/
-batch_normalization_5/AssignMovingAvg_1/sub/x�
+batch_normalization_5/AssignMovingAvg_1/subSub6batch_normalization_5/AssignMovingAvg_1/sub/x:output:0&batch_normalization_5/Const_2:output:0*
T0*@
_class6
42loc:@batch_normalization_5/AssignMovingAvg_1/25319*
_output_shapes
: 2-
+batch_normalization_5/AssignMovingAvg_1/sub�
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_5_assignmovingavg_1_25319*
_output_shapes
:@*
dtype028
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp�
-batch_normalization_5/AssignMovingAvg_1/sub_1Sub>batch_normalization_5/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_5/FusedBatchNormV3:batch_variance:0*
T0*@
_class6
42loc:@batch_normalization_5/AssignMovingAvg_1/25319*
_output_shapes
:@2/
-batch_normalization_5/AssignMovingAvg_1/sub_1�
+batch_normalization_5/AssignMovingAvg_1/mulMul1batch_normalization_5/AssignMovingAvg_1/sub_1:z:0/batch_normalization_5/AssignMovingAvg_1/sub:z:0*
T0*@
_class6
42loc:@batch_normalization_5/AssignMovingAvg_1/25319*
_output_shapes
:@2-
+batch_normalization_5/AssignMovingAvg_1/mul�
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_5_assignmovingavg_1_25319/batch_normalization_5/AssignMovingAvg_1/mul:z:07^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp*@
_class6
42loc:@batch_normalization_5/AssignMovingAvg_1/25319*
_output_shapes
 *
dtype02=
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp�
	add_2/addAddV2*batch_normalization_5/FusedBatchNormV3:y:0"average_pooling2d/AvgPool:output:0*
T0*/
_output_shapes
:���������@2
	add_2/add�
average_pooling2d_1/AvgPoolAvgPooladd_2/add:z:0*
T0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
average_pooling2d_1/AvgPool�
'res_conv_A_repr_d/Conv2D/ReadVariableOpReadVariableOp0res_conv_a_repr_d_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'res_conv_A_repr_d/Conv2D/ReadVariableOp�
res_conv_A_repr_d/Conv2DConv2D$average_pooling2d_1/AvgPool:output:0/res_conv_A_repr_d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
res_conv_A_repr_d/Conv2D�
(res_conv_A_repr_d/BiasAdd/ReadVariableOpReadVariableOp1res_conv_a_repr_d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(res_conv_A_repr_d/BiasAdd/ReadVariableOp�
res_conv_A_repr_d/BiasAddBiasAdd!res_conv_A_repr_d/Conv2D:output:00res_conv_A_repr_d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
res_conv_A_repr_d/BiasAdd�
activation_6/ReluRelu"res_conv_A_repr_d/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
activation_6/Relu�
"batch_normalization_6/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_6/LogicalAnd/x�
"batch_normalization_6/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_6/LogicalAnd/y�
 batch_normalization_6/LogicalAnd
LogicalAnd+batch_normalization_6/LogicalAnd/x:output:0+batch_normalization_6/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_6/LogicalAnd�
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_6/ReadVariableOp�
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_6/ReadVariableOp_1}
batch_normalization_6/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_6/Const�
batch_normalization_6/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_6/Const_1�
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3activation_6/Relu:activations:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0$batch_normalization_6/Const:output:0&batch_normalization_6/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2(
&batch_normalization_6/FusedBatchNormV3�
batch_normalization_6/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_6/Const_2�
+batch_normalization_6/AssignMovingAvg/sub/xConst*>
_class4
20loc:@batch_normalization_6/AssignMovingAvg/25351*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+batch_normalization_6/AssignMovingAvg/sub/x�
)batch_normalization_6/AssignMovingAvg/subSub4batch_normalization_6/AssignMovingAvg/sub/x:output:0&batch_normalization_6/Const_2:output:0*
T0*>
_class4
20loc:@batch_normalization_6/AssignMovingAvg/25351*
_output_shapes
: 2+
)batch_normalization_6/AssignMovingAvg/sub�
4batch_normalization_6/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_6_assignmovingavg_25351*
_output_shapes
:@*
dtype026
4batch_normalization_6/AssignMovingAvg/ReadVariableOp�
+batch_normalization_6/AssignMovingAvg/sub_1Sub<batch_normalization_6/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_6/FusedBatchNormV3:batch_mean:0*
T0*>
_class4
20loc:@batch_normalization_6/AssignMovingAvg/25351*
_output_shapes
:@2-
+batch_normalization_6/AssignMovingAvg/sub_1�
)batch_normalization_6/AssignMovingAvg/mulMul/batch_normalization_6/AssignMovingAvg/sub_1:z:0-batch_normalization_6/AssignMovingAvg/sub:z:0*
T0*>
_class4
20loc:@batch_normalization_6/AssignMovingAvg/25351*
_output_shapes
:@2+
)batch_normalization_6/AssignMovingAvg/mul�
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_6_assignmovingavg_25351-batch_normalization_6/AssignMovingAvg/mul:z:05^batch_normalization_6/AssignMovingAvg/ReadVariableOp*>
_class4
20loc:@batch_normalization_6/AssignMovingAvg/25351*
_output_shapes
 *
dtype02;
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_6/AssignMovingAvg_1/sub/xConst*@
_class6
42loc:@batch_normalization_6/AssignMovingAvg_1/25358*
_output_shapes
: *
dtype0*
valueB
 *  �?2/
-batch_normalization_6/AssignMovingAvg_1/sub/x�
+batch_normalization_6/AssignMovingAvg_1/subSub6batch_normalization_6/AssignMovingAvg_1/sub/x:output:0&batch_normalization_6/Const_2:output:0*
T0*@
_class6
42loc:@batch_normalization_6/AssignMovingAvg_1/25358*
_output_shapes
: 2-
+batch_normalization_6/AssignMovingAvg_1/sub�
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_6_assignmovingavg_1_25358*
_output_shapes
:@*
dtype028
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp�
-batch_normalization_6/AssignMovingAvg_1/sub_1Sub>batch_normalization_6/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_6/FusedBatchNormV3:batch_variance:0*
T0*@
_class6
42loc:@batch_normalization_6/AssignMovingAvg_1/25358*
_output_shapes
:@2/
-batch_normalization_6/AssignMovingAvg_1/sub_1�
+batch_normalization_6/AssignMovingAvg_1/mulMul1batch_normalization_6/AssignMovingAvg_1/sub_1:z:0/batch_normalization_6/AssignMovingAvg_1/sub:z:0*
T0*@
_class6
42loc:@batch_normalization_6/AssignMovingAvg_1/25358*
_output_shapes
:@2-
+batch_normalization_6/AssignMovingAvg_1/mul�
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_6_assignmovingavg_1_25358/batch_normalization_6/AssignMovingAvg_1/mul:z:07^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp*@
_class6
42loc:@batch_normalization_6/AssignMovingAvg_1/25358*
_output_shapes
 *
dtype02=
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp�
'res_conv_B_repr_d/Conv2D/ReadVariableOpReadVariableOp0res_conv_b_repr_d_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'res_conv_B_repr_d/Conv2D/ReadVariableOp�
res_conv_B_repr_d/Conv2DConv2D*batch_normalization_6/FusedBatchNormV3:y:0/res_conv_B_repr_d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
res_conv_B_repr_d/Conv2D�
(res_conv_B_repr_d/BiasAdd/ReadVariableOpReadVariableOp1res_conv_b_repr_d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(res_conv_B_repr_d/BiasAdd/ReadVariableOp�
res_conv_B_repr_d/BiasAddBiasAdd!res_conv_B_repr_d/Conv2D:output:00res_conv_B_repr_d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
res_conv_B_repr_d/BiasAdd�
activation_7/ReluRelu"res_conv_B_repr_d/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
activation_7/Relu�
"batch_normalization_7/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_7/LogicalAnd/x�
"batch_normalization_7/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_7/LogicalAnd/y�
 batch_normalization_7/LogicalAnd
LogicalAnd+batch_normalization_7/LogicalAnd/x:output:0+batch_normalization_7/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_7/LogicalAnd�
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_7/ReadVariableOp�
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_7/ReadVariableOp_1}
batch_normalization_7/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_7/Const�
batch_normalization_7/Const_1Const*
_output_shapes
: *
dtype0*
valueB 2
batch_normalization_7/Const_1�
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3activation_7/Relu:activations:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0$batch_normalization_7/Const:output:0&batch_normalization_7/Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2(
&batch_normalization_7/FusedBatchNormV3�
batch_normalization_7/Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_7/Const_2�
+batch_normalization_7/AssignMovingAvg/sub/xConst*>
_class4
20loc:@batch_normalization_7/AssignMovingAvg/25388*
_output_shapes
: *
dtype0*
valueB
 *  �?2-
+batch_normalization_7/AssignMovingAvg/sub/x�
)batch_normalization_7/AssignMovingAvg/subSub4batch_normalization_7/AssignMovingAvg/sub/x:output:0&batch_normalization_7/Const_2:output:0*
T0*>
_class4
20loc:@batch_normalization_7/AssignMovingAvg/25388*
_output_shapes
: 2+
)batch_normalization_7/AssignMovingAvg/sub�
4batch_normalization_7/AssignMovingAvg/ReadVariableOpReadVariableOp+batch_normalization_7_assignmovingavg_25388*
_output_shapes
:@*
dtype026
4batch_normalization_7/AssignMovingAvg/ReadVariableOp�
+batch_normalization_7/AssignMovingAvg/sub_1Sub<batch_normalization_7/AssignMovingAvg/ReadVariableOp:value:03batch_normalization_7/FusedBatchNormV3:batch_mean:0*
T0*>
_class4
20loc:@batch_normalization_7/AssignMovingAvg/25388*
_output_shapes
:@2-
+batch_normalization_7/AssignMovingAvg/sub_1�
)batch_normalization_7/AssignMovingAvg/mulMul/batch_normalization_7/AssignMovingAvg/sub_1:z:0-batch_normalization_7/AssignMovingAvg/sub:z:0*
T0*>
_class4
20loc:@batch_normalization_7/AssignMovingAvg/25388*
_output_shapes
:@2+
)batch_normalization_7/AssignMovingAvg/mul�
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOpAssignSubVariableOp+batch_normalization_7_assignmovingavg_25388-batch_normalization_7/AssignMovingAvg/mul:z:05^batch_normalization_7/AssignMovingAvg/ReadVariableOp*>
_class4
20loc:@batch_normalization_7/AssignMovingAvg/25388*
_output_shapes
 *
dtype02;
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp�
-batch_normalization_7/AssignMovingAvg_1/sub/xConst*@
_class6
42loc:@batch_normalization_7/AssignMovingAvg_1/25395*
_output_shapes
: *
dtype0*
valueB
 *  �?2/
-batch_normalization_7/AssignMovingAvg_1/sub/x�
+batch_normalization_7/AssignMovingAvg_1/subSub6batch_normalization_7/AssignMovingAvg_1/sub/x:output:0&batch_normalization_7/Const_2:output:0*
T0*@
_class6
42loc:@batch_normalization_7/AssignMovingAvg_1/25395*
_output_shapes
: 2-
+batch_normalization_7/AssignMovingAvg_1/sub�
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOpReadVariableOp-batch_normalization_7_assignmovingavg_1_25395*
_output_shapes
:@*
dtype028
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp�
-batch_normalization_7/AssignMovingAvg_1/sub_1Sub>batch_normalization_7/AssignMovingAvg_1/ReadVariableOp:value:07batch_normalization_7/FusedBatchNormV3:batch_variance:0*
T0*@
_class6
42loc:@batch_normalization_7/AssignMovingAvg_1/25395*
_output_shapes
:@2/
-batch_normalization_7/AssignMovingAvg_1/sub_1�
+batch_normalization_7/AssignMovingAvg_1/mulMul1batch_normalization_7/AssignMovingAvg_1/sub_1:z:0/batch_normalization_7/AssignMovingAvg_1/sub:z:0*
T0*@
_class6
42loc:@batch_normalization_7/AssignMovingAvg_1/25395*
_output_shapes
:@2-
+batch_normalization_7/AssignMovingAvg_1/mul�
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOp-batch_normalization_7_assignmovingavg_1_25395/batch_normalization_7/AssignMovingAvg_1/mul:z:07^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp*@
_class6
42loc:@batch_normalization_7/AssignMovingAvg_1/25395*
_output_shapes
 *
dtype02=
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp�
	add_3/addAddV2*batch_normalization_7/FusedBatchNormV3:y:0$average_pooling2d_1/AvgPool:output:0*
T0*/
_output_shapes
:���������@2
	add_3/add�
 repr_board/Conv2D/ReadVariableOpReadVariableOp)repr_board_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02"
 repr_board/Conv2D/ReadVariableOp�
repr_board/Conv2DConv2Dadd_3/add:z:0(repr_board/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
2
repr_board/Conv2D�
!repr_board/BiasAdd/ReadVariableOpReadVariableOp*repr_board_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!repr_board/BiasAdd/ReadVariableOp�
repr_board/BiasAddBiasAddrepr_board/Conv2D:output:0)repr_board/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
repr_board/BiasAdd�
repr_board/ReluRelurepr_board/BiasAdd:output:0*
T0*/
_output_shapes
:���������2
repr_board/Relu�
:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_a_repr_a_conv2d_readvariableop_resource(^res_conv_A_repr_a/Conv2D/ReadVariableOp*&
_output_shapes
:  *
dtype02<
:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_a/kernel/Regularizer/SquareSquareBres_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2-
+res_conv_A_repr_a/kernel/Regularizer/Square�
*res_conv_A_repr_a/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_a/kernel/Regularizer/Const�
(res_conv_A_repr_a/kernel/Regularizer/SumSum/res_conv_A_repr_a/kernel/Regularizer/Square:y:03res_conv_A_repr_a/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_a/kernel/Regularizer/Sum�
*res_conv_A_repr_a/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_a/kernel/Regularizer/mul/x�
(res_conv_A_repr_a/kernel/Regularizer/mulMul3res_conv_A_repr_a/kernel/Regularizer/mul/x:output:01res_conv_A_repr_a/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_a/kernel/Regularizer/mul�
*res_conv_A_repr_a/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_a/kernel/Regularizer/add/x�
(res_conv_A_repr_a/kernel/Regularizer/addAddV23res_conv_A_repr_a/kernel/Regularizer/add/x:output:0,res_conv_A_repr_a/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_a/kernel/Regularizer/add�
:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_b_repr_a_conv2d_readvariableop_resource(^res_conv_B_repr_a/Conv2D/ReadVariableOp*&
_output_shapes
:  *
dtype02<
:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_a/kernel/Regularizer/SquareSquareBres_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2-
+res_conv_B_repr_a/kernel/Regularizer/Square�
*res_conv_B_repr_a/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_a/kernel/Regularizer/Const�
(res_conv_B_repr_a/kernel/Regularizer/SumSum/res_conv_B_repr_a/kernel/Regularizer/Square:y:03res_conv_B_repr_a/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_a/kernel/Regularizer/Sum�
*res_conv_B_repr_a/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_a/kernel/Regularizer/mul/x�
(res_conv_B_repr_a/kernel/Regularizer/mulMul3res_conv_B_repr_a/kernel/Regularizer/mul/x:output:01res_conv_B_repr_a/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_a/kernel/Regularizer/mul�
*res_conv_B_repr_a/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_a/kernel/Regularizer/add/x�
(res_conv_B_repr_a/kernel/Regularizer/addAddV23res_conv_B_repr_a/kernel/Regularizer/add/x:output:0,res_conv_B_repr_a/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_a/kernel/Regularizer/add�
:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_a_repr_b_conv2d_readvariableop_resource(^res_conv_A_repr_b/Conv2D/ReadVariableOp*&
_output_shapes
:@@*
dtype02<
:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_b/kernel/Regularizer/SquareSquareBres_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_A_repr_b/kernel/Regularizer/Square�
*res_conv_A_repr_b/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_b/kernel/Regularizer/Const�
(res_conv_A_repr_b/kernel/Regularizer/SumSum/res_conv_A_repr_b/kernel/Regularizer/Square:y:03res_conv_A_repr_b/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_b/kernel/Regularizer/Sum�
*res_conv_A_repr_b/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_b/kernel/Regularizer/mul/x�
(res_conv_A_repr_b/kernel/Regularizer/mulMul3res_conv_A_repr_b/kernel/Regularizer/mul/x:output:01res_conv_A_repr_b/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_b/kernel/Regularizer/mul�
*res_conv_A_repr_b/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_b/kernel/Regularizer/add/x�
(res_conv_A_repr_b/kernel/Regularizer/addAddV23res_conv_A_repr_b/kernel/Regularizer/add/x:output:0,res_conv_A_repr_b/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_b/kernel/Regularizer/add�
:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_b_repr_b_conv2d_readvariableop_resource(^res_conv_B_repr_b/Conv2D/ReadVariableOp*&
_output_shapes
:@@*
dtype02<
:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_b/kernel/Regularizer/SquareSquareBres_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_B_repr_b/kernel/Regularizer/Square�
*res_conv_B_repr_b/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_b/kernel/Regularizer/Const�
(res_conv_B_repr_b/kernel/Regularizer/SumSum/res_conv_B_repr_b/kernel/Regularizer/Square:y:03res_conv_B_repr_b/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_b/kernel/Regularizer/Sum�
*res_conv_B_repr_b/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_b/kernel/Regularizer/mul/x�
(res_conv_B_repr_b/kernel/Regularizer/mulMul3res_conv_B_repr_b/kernel/Regularizer/mul/x:output:01res_conv_B_repr_b/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_b/kernel/Regularizer/mul�
*res_conv_B_repr_b/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_b/kernel/Regularizer/add/x�
(res_conv_B_repr_b/kernel/Regularizer/addAddV23res_conv_B_repr_b/kernel/Regularizer/add/x:output:0,res_conv_B_repr_b/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_b/kernel/Regularizer/add�
:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_a_repr_c_conv2d_readvariableop_resource(^res_conv_A_repr_c/Conv2D/ReadVariableOp*&
_output_shapes
:@@*
dtype02<
:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_c/kernel/Regularizer/SquareSquareBres_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_A_repr_c/kernel/Regularizer/Square�
*res_conv_A_repr_c/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_c/kernel/Regularizer/Const�
(res_conv_A_repr_c/kernel/Regularizer/SumSum/res_conv_A_repr_c/kernel/Regularizer/Square:y:03res_conv_A_repr_c/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_c/kernel/Regularizer/Sum�
*res_conv_A_repr_c/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_c/kernel/Regularizer/mul/x�
(res_conv_A_repr_c/kernel/Regularizer/mulMul3res_conv_A_repr_c/kernel/Regularizer/mul/x:output:01res_conv_A_repr_c/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_c/kernel/Regularizer/mul�
*res_conv_A_repr_c/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_c/kernel/Regularizer/add/x�
(res_conv_A_repr_c/kernel/Regularizer/addAddV23res_conv_A_repr_c/kernel/Regularizer/add/x:output:0,res_conv_A_repr_c/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_c/kernel/Regularizer/add�
:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_b_repr_c_conv2d_readvariableop_resource(^res_conv_B_repr_c/Conv2D/ReadVariableOp*&
_output_shapes
:@@*
dtype02<
:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_c/kernel/Regularizer/SquareSquareBres_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_B_repr_c/kernel/Regularizer/Square�
*res_conv_B_repr_c/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_c/kernel/Regularizer/Const�
(res_conv_B_repr_c/kernel/Regularizer/SumSum/res_conv_B_repr_c/kernel/Regularizer/Square:y:03res_conv_B_repr_c/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_c/kernel/Regularizer/Sum�
*res_conv_B_repr_c/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_c/kernel/Regularizer/mul/x�
(res_conv_B_repr_c/kernel/Regularizer/mulMul3res_conv_B_repr_c/kernel/Regularizer/mul/x:output:01res_conv_B_repr_c/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_c/kernel/Regularizer/mul�
*res_conv_B_repr_c/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_c/kernel/Regularizer/add/x�
(res_conv_B_repr_c/kernel/Regularizer/addAddV23res_conv_B_repr_c/kernel/Regularizer/add/x:output:0,res_conv_B_repr_c/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_c/kernel/Regularizer/add�
:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_a_repr_d_conv2d_readvariableop_resource(^res_conv_A_repr_d/Conv2D/ReadVariableOp*&
_output_shapes
:@@*
dtype02<
:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_d/kernel/Regularizer/SquareSquareBres_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_A_repr_d/kernel/Regularizer/Square�
*res_conv_A_repr_d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_d/kernel/Regularizer/Const�
(res_conv_A_repr_d/kernel/Regularizer/SumSum/res_conv_A_repr_d/kernel/Regularizer/Square:y:03res_conv_A_repr_d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_d/kernel/Regularizer/Sum�
*res_conv_A_repr_d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_d/kernel/Regularizer/mul/x�
(res_conv_A_repr_d/kernel/Regularizer/mulMul3res_conv_A_repr_d/kernel/Regularizer/mul/x:output:01res_conv_A_repr_d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_d/kernel/Regularizer/mul�
*res_conv_A_repr_d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_d/kernel/Regularizer/add/x�
(res_conv_A_repr_d/kernel/Regularizer/addAddV23res_conv_A_repr_d/kernel/Regularizer/add/x:output:0,res_conv_A_repr_d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_d/kernel/Regularizer/add�
:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_b_repr_d_conv2d_readvariableop_resource(^res_conv_B_repr_d/Conv2D/ReadVariableOp*&
_output_shapes
:@@*
dtype02<
:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_d/kernel/Regularizer/SquareSquareBres_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_B_repr_d/kernel/Regularizer/Square�
*res_conv_B_repr_d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_d/kernel/Regularizer/Const�
(res_conv_B_repr_d/kernel/Regularizer/SumSum/res_conv_B_repr_d/kernel/Regularizer/Square:y:03res_conv_B_repr_d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_d/kernel/Regularizer/Sum�
*res_conv_B_repr_d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_d/kernel/Regularizer/mul/x�
(res_conv_B_repr_d/kernel/Regularizer/mulMul3res_conv_B_repr_d/kernel/Regularizer/mul/x:output:01res_conv_B_repr_d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_d/kernel/Regularizer/mul�
*res_conv_B_repr_d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_d/kernel/Regularizer/add/x�
(res_conv_B_repr_d/kernel/Regularizer/addAddV23res_conv_B_repr_d/kernel/Regularizer/add/x:output:0,res_conv_B_repr_d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_d/kernel/Regularizer/add�
3repr_board/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)repr_board_conv2d_readvariableop_resource!^repr_board/Conv2D/ReadVariableOp*&
_output_shapes
:@*
dtype025
3repr_board/kernel/Regularizer/Square/ReadVariableOp�
$repr_board/kernel/Regularizer/SquareSquare;repr_board/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2&
$repr_board/kernel/Regularizer/Square�
#repr_board/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2%
#repr_board/kernel/Regularizer/Const�
!repr_board/kernel/Regularizer/SumSum(repr_board/kernel/Regularizer/Square:y:0,repr_board/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!repr_board/kernel/Regularizer/Sum�
#repr_board/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72%
#repr_board/kernel/Regularizer/mul/x�
!repr_board/kernel/Regularizer/mulMul,repr_board/kernel/Regularizer/mul/x:output:0*repr_board/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!repr_board/kernel/Regularizer/mul�
#repr_board/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#repr_board/kernel/Regularizer/add/x�
!repr_board/kernel/Regularizer/addAddV2,repr_board/kernel/Regularizer/add/x:output:0%repr_board/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2#
!repr_board/kernel/Regularizer/add�
1repr_board/bias/Regularizer/Square/ReadVariableOpReadVariableOp*repr_board_biasadd_readvariableop_resource"^repr_board/BiasAdd/ReadVariableOp*
_output_shapes
:*
dtype023
1repr_board/bias/Regularizer/Square/ReadVariableOp�
"repr_board/bias/Regularizer/SquareSquare9repr_board/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2$
"repr_board/bias/Regularizer/Square�
!repr_board/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!repr_board/bias/Regularizer/Const�
repr_board/bias/Regularizer/SumSum&repr_board/bias/Regularizer/Square:y:0*repr_board/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
repr_board/bias/Regularizer/Sum�
!repr_board/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72#
!repr_board/bias/Regularizer/mul/x�
repr_board/bias/Regularizer/mulMul*repr_board/bias/Regularizer/mul/x:output:0(repr_board/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
repr_board/bias/Regularizer/mul�
!repr_board/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!repr_board/bias/Regularizer/add/x�
repr_board/bias/Regularizer/addAddV2*repr_board/bias/Regularizer/add/x:output:0#repr_board/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
repr_board/bias/Regularizer/add� 
IdentityIdentityrepr_board/Relu:activations:08^batch_normalization/AssignMovingAvg/AssignSubVariableOp3^batch_normalization/AssignMovingAvg/ReadVariableOp:^batch_normalization/AssignMovingAvg_1/AssignSubVariableOp5^batch_normalization/AssignMovingAvg_1/ReadVariableOp#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_1:^batch_normalization_1/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_1/AssignMovingAvg/ReadVariableOp<^batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_1/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_1:^batch_normalization_2/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_2/AssignMovingAvg/ReadVariableOp<^batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_2/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_1:^batch_normalization_3/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_3/AssignMovingAvg/ReadVariableOp<^batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_3/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_1:^batch_normalization_4/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_4/AssignMovingAvg/ReadVariableOp<^batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_4/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_1:^batch_normalization_5/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_5/AssignMovingAvg/ReadVariableOp<^batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_5/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_1:^batch_normalization_6/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_6/AssignMovingAvg/ReadVariableOp<^batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_6/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_1:^batch_normalization_7/AssignMovingAvg/AssignSubVariableOp5^batch_normalization_7/AssignMovingAvg/ReadVariableOp<^batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp7^batch_normalization_7/AssignMovingAvg_1/ReadVariableOp%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp"^repr_board/BiasAdd/ReadVariableOp!^repr_board/Conv2D/ReadVariableOp2^repr_board/bias/Regularizer/Square/ReadVariableOp4^repr_board/kernel/Regularizer/Square/ReadVariableOp)^res_conv_A_repr_a/BiasAdd/ReadVariableOp(^res_conv_A_repr_a/Conv2D/ReadVariableOp;^res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp)^res_conv_A_repr_b/BiasAdd/ReadVariableOp(^res_conv_A_repr_b/Conv2D/ReadVariableOp;^res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp)^res_conv_A_repr_c/BiasAdd/ReadVariableOp(^res_conv_A_repr_c/Conv2D/ReadVariableOp;^res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp)^res_conv_A_repr_d/BiasAdd/ReadVariableOp(^res_conv_A_repr_d/Conv2D/ReadVariableOp;^res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp)^res_conv_B_repr_a/BiasAdd/ReadVariableOp(^res_conv_B_repr_a/Conv2D/ReadVariableOp;^res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp)^res_conv_B_repr_b/BiasAdd/ReadVariableOp(^res_conv_B_repr_b/Conv2D/ReadVariableOp;^res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp)^res_conv_B_repr_c/BiasAdd/ReadVariableOp(^res_conv_B_repr_c/Conv2D/ReadVariableOp;^res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp)^res_conv_B_repr_d/BiasAdd/ReadVariableOp(^res_conv_B_repr_d/Conv2D/ReadVariableOp;^res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������``::::::::::::::::::::::::::::::::::::::::::::::::::::::2r
7batch_normalization/AssignMovingAvg/AssignSubVariableOp7batch_normalization/AssignMovingAvg/AssignSubVariableOp2h
2batch_normalization/AssignMovingAvg/ReadVariableOp2batch_normalization/AssignMovingAvg/ReadVariableOp2v
9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp9batch_normalization/AssignMovingAvg_1/AssignSubVariableOp2l
4batch_normalization/AssignMovingAvg_1/ReadVariableOp4batch_normalization/AssignMovingAvg_1/ReadVariableOp2H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12v
9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp9batch_normalization_1/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_1/AssignMovingAvg/ReadVariableOp4batch_normalization_1/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_1/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp6batch_normalization_1/AssignMovingAvg_1/ReadVariableOp2L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12v
9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp9batch_normalization_2/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_2/AssignMovingAvg/ReadVariableOp4batch_normalization_2/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_2/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp6batch_normalization_2/AssignMovingAvg_1/ReadVariableOp2L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12v
9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp9batch_normalization_3/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_3/AssignMovingAvg/ReadVariableOp4batch_normalization_3/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_3/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp6batch_normalization_3/AssignMovingAvg_1/ReadVariableOp2L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12v
9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp9batch_normalization_4/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_4/AssignMovingAvg/ReadVariableOp4batch_normalization_4/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_4/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp6batch_normalization_4/AssignMovingAvg_1/ReadVariableOp2L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12v
9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp9batch_normalization_5/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_5/AssignMovingAvg/ReadVariableOp4batch_normalization_5/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_5/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp6batch_normalization_5/AssignMovingAvg_1/ReadVariableOp2L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12v
9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp9batch_normalization_6/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_6/AssignMovingAvg/ReadVariableOp4batch_normalization_6/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_6/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp6batch_normalization_6/AssignMovingAvg_1/ReadVariableOp2L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12v
9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp9batch_normalization_7/AssignMovingAvg/AssignSubVariableOp2l
4batch_normalization_7/AssignMovingAvg/ReadVariableOp4batch_normalization_7/AssignMovingAvg/ReadVariableOp2z
;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp;batch_normalization_7/AssignMovingAvg_1/AssignSubVariableOp2p
6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp6batch_normalization_7/AssignMovingAvg_1/ReadVariableOp2L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2F
!repr_board/BiasAdd/ReadVariableOp!repr_board/BiasAdd/ReadVariableOp2D
 repr_board/Conv2D/ReadVariableOp repr_board/Conv2D/ReadVariableOp2f
1repr_board/bias/Regularizer/Square/ReadVariableOp1repr_board/bias/Regularizer/Square/ReadVariableOp2j
3repr_board/kernel/Regularizer/Square/ReadVariableOp3repr_board/kernel/Regularizer/Square/ReadVariableOp2T
(res_conv_A_repr_a/BiasAdd/ReadVariableOp(res_conv_A_repr_a/BiasAdd/ReadVariableOp2R
'res_conv_A_repr_a/Conv2D/ReadVariableOp'res_conv_A_repr_a/Conv2D/ReadVariableOp2x
:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp2T
(res_conv_A_repr_b/BiasAdd/ReadVariableOp(res_conv_A_repr_b/BiasAdd/ReadVariableOp2R
'res_conv_A_repr_b/Conv2D/ReadVariableOp'res_conv_A_repr_b/Conv2D/ReadVariableOp2x
:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp2T
(res_conv_A_repr_c/BiasAdd/ReadVariableOp(res_conv_A_repr_c/BiasAdd/ReadVariableOp2R
'res_conv_A_repr_c/Conv2D/ReadVariableOp'res_conv_A_repr_c/Conv2D/ReadVariableOp2x
:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp2T
(res_conv_A_repr_d/BiasAdd/ReadVariableOp(res_conv_A_repr_d/BiasAdd/ReadVariableOp2R
'res_conv_A_repr_d/Conv2D/ReadVariableOp'res_conv_A_repr_d/Conv2D/ReadVariableOp2x
:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp2T
(res_conv_B_repr_a/BiasAdd/ReadVariableOp(res_conv_B_repr_a/BiasAdd/ReadVariableOp2R
'res_conv_B_repr_a/Conv2D/ReadVariableOp'res_conv_B_repr_a/Conv2D/ReadVariableOp2x
:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp2T
(res_conv_B_repr_b/BiasAdd/ReadVariableOp(res_conv_B_repr_b/BiasAdd/ReadVariableOp2R
'res_conv_B_repr_b/Conv2D/ReadVariableOp'res_conv_B_repr_b/Conv2D/ReadVariableOp2x
:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp2T
(res_conv_B_repr_c/BiasAdd/ReadVariableOp(res_conv_B_repr_c/BiasAdd/ReadVariableOp2R
'res_conv_B_repr_c/Conv2D/ReadVariableOp'res_conv_B_repr_c/Conv2D/ReadVariableOp2x
:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp2T
(res_conv_B_repr_d/BiasAdd/ReadVariableOp(res_conv_B_repr_d/BiasAdd/ReadVariableOp2R
'res_conv_B_repr_d/Conv2D/ReadVariableOp'res_conv_B_repr_d/Conv2D/ReadVariableOp2x
:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26876

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_3_layer_call_fn_26704

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_225162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_24063

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�$
�
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_22989

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_22974
assignmovingavg_1_22981
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/22974*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/22974*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_22974*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/22974*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/22974*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_22974AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/22974*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/22981*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/22981*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_22981*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/22981*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/22981*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_22981AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/22981*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
O
#__inference_add_layer_call_fn_26348
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_234732
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������00 2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������00 :���������00 :( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
�
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_27422

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
c
G__inference_activation_5_layer_call_and_return_conditional_losses_26907

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�#
�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_23534

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_23519
assignmovingavg_1_23526
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/23519*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/23519*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23519*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/23519*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/23519*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23519AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/23519*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/23526*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/23526*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23526*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/23526*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/23526*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23526AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/23526*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�#
�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_23629

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_23614
assignmovingavg_1_23621
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/23614*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/23614*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23614*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/23614*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/23614*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23614AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/23614*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/23621*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/23621*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23621*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/23621*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/23621*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23621AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/23621*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
`
D__inference_reshape_1_layer_call_and_return_conditional_losses_23266

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :`2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :`2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/3�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapew
ReshapeReshapeinputsReshape/shape:output:0*
T0*/
_output_shapes
:���������``2	
Reshapel
IdentityIdentityReshape:output:0*
T0*/
_output_shapes
:���������``2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������``:& "
 
_user_specified_nameinputs
�
�
3__inference_batch_normalization_layer_call_fn_26084

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_233482
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������00 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������00 ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
Q
%__inference_add_3_layer_call_fn_27452
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*I
fDRB
@__inference_add_3_layer_call_and_return_conditional_losses_240932
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������@:���������@:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
�
5__inference_batch_normalization_5_layer_call_fn_27063

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_228172
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
1__inference_res_conv_A_repr_a_layer_call_fn_21891

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+��������������������������� */
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_A_repr_a_layer_call_and_return_conditional_losses_218832
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�%
�
.__inference_Representation_layer_call_fn_24594	
board"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7"
statefulpartitionedcall_args_8"
statefulpartitionedcall_args_9#
statefulpartitionedcall_args_10#
statefulpartitionedcall_args_11#
statefulpartitionedcall_args_12#
statefulpartitionedcall_args_13#
statefulpartitionedcall_args_14#
statefulpartitionedcall_args_15#
statefulpartitionedcall_args_16#
statefulpartitionedcall_args_17#
statefulpartitionedcall_args_18#
statefulpartitionedcall_args_19#
statefulpartitionedcall_args_20#
statefulpartitionedcall_args_21#
statefulpartitionedcall_args_22#
statefulpartitionedcall_args_23#
statefulpartitionedcall_args_24#
statefulpartitionedcall_args_25#
statefulpartitionedcall_args_26#
statefulpartitionedcall_args_27#
statefulpartitionedcall_args_28#
statefulpartitionedcall_args_29#
statefulpartitionedcall_args_30#
statefulpartitionedcall_args_31#
statefulpartitionedcall_args_32#
statefulpartitionedcall_args_33#
statefulpartitionedcall_args_34#
statefulpartitionedcall_args_35#
statefulpartitionedcall_args_36#
statefulpartitionedcall_args_37#
statefulpartitionedcall_args_38#
statefulpartitionedcall_args_39#
statefulpartitionedcall_args_40#
statefulpartitionedcall_args_41#
statefulpartitionedcall_args_42#
statefulpartitionedcall_args_43#
statefulpartitionedcall_args_44#
statefulpartitionedcall_args_45#
statefulpartitionedcall_args_46#
statefulpartitionedcall_args_47#
statefulpartitionedcall_args_48#
statefulpartitionedcall_args_49#
statefulpartitionedcall_args_50#
statefulpartitionedcall_args_51#
statefulpartitionedcall_args_52#
statefulpartitionedcall_args_53#
statefulpartitionedcall_args_54
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallboardstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7statefulpartitionedcall_args_8statefulpartitionedcall_args_9statefulpartitionedcall_args_10statefulpartitionedcall_args_11statefulpartitionedcall_args_12statefulpartitionedcall_args_13statefulpartitionedcall_args_14statefulpartitionedcall_args_15statefulpartitionedcall_args_16statefulpartitionedcall_args_17statefulpartitionedcall_args_18statefulpartitionedcall_args_19statefulpartitionedcall_args_20statefulpartitionedcall_args_21statefulpartitionedcall_args_22statefulpartitionedcall_args_23statefulpartitionedcall_args_24statefulpartitionedcall_args_25statefulpartitionedcall_args_26statefulpartitionedcall_args_27statefulpartitionedcall_args_28statefulpartitionedcall_args_29statefulpartitionedcall_args_30statefulpartitionedcall_args_31statefulpartitionedcall_args_32statefulpartitionedcall_args_33statefulpartitionedcall_args_34statefulpartitionedcall_args_35statefulpartitionedcall_args_36statefulpartitionedcall_args_37statefulpartitionedcall_args_38statefulpartitionedcall_args_39statefulpartitionedcall_args_40statefulpartitionedcall_args_41statefulpartitionedcall_args_42statefulpartitionedcall_args_43statefulpartitionedcall_args_44statefulpartitionedcall_args_45statefulpartitionedcall_args_46statefulpartitionedcall_args_47statefulpartitionedcall_args_48statefulpartitionedcall_args_49statefulpartitionedcall_args_50statefulpartitionedcall_args_51statefulpartitionedcall_args_52statefulpartitionedcall_args_53statefulpartitionedcall_args_54*B
Tin;
927*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������*/
config_proto

GPU

CPU2 *0J 8*R
fMRK
I__inference_Representation_layer_call_and_return_conditional_losses_245372
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������``::::::::::::::::::::::::::::::::::::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:% !

_user_specified_nameboard
�
F
*__inference_activation_layer_call_fn_25998

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_232852
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������00 2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������00 :& "
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_2_layer_call_fn_26452

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_223562
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_1_layer_call_fn_26262

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_234432
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������00 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������00 ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�#
�
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_27222

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_27207
assignmovingavg_1_27214
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/27207*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/27207*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_27207*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/27207*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/27207*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_27207AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/27207*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/27214*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/27214*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_27214*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/27214*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/27214*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_27214AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/27214*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
1__inference_res_conv_B_repr_b_layer_call_fn_22391

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_B_repr_b_layer_call_and_return_conditional_losses_223832
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26508

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_3_layer_call_fn_26621

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_236292
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_22516

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
__inference_loss_fn_6_27559G
Cres_conv_a_repr_d_kernel_regularizer_square_readvariableop_resource
identity��:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp�
:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpCres_conv_a_repr_d_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@@*
dtype02<
:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_d/kernel/Regularizer/SquareSquareBres_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_A_repr_d/kernel/Regularizer/Square�
*res_conv_A_repr_d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_d/kernel/Regularizer/Const�
(res_conv_A_repr_d/kernel/Regularizer/SumSum/res_conv_A_repr_d/kernel/Regularizer/Square:y:03res_conv_A_repr_d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_d/kernel/Regularizer/Sum�
*res_conv_A_repr_d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_d/kernel/Regularizer/mul/x�
(res_conv_A_repr_d/kernel/Regularizer/mulMul3res_conv_A_repr_d/kernel/Regularizer/mul/x:output:01res_conv_A_repr_d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_d/kernel/Regularizer/mul�
*res_conv_A_repr_d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_d/kernel/Regularizer/add/x�
(res_conv_A_repr_d/kernel/Regularizer/addAddV23res_conv_A_repr_d/kernel/Regularizer/add/x:output:0,res_conv_A_repr_d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_d/kernel/Regularizer/add�
IdentityIdentity,res_conv_A_repr_d/kernel/Regularizer/add:z:0;^res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2x
:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp
�
�
5__inference_batch_normalization_6_layer_call_fn_27188

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_230202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_3_layer_call_fn_26630

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_236512
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�d
�
__inference__traced_save_27784
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop7
3savev2_res_conv_a_repr_a_kernel_read_readvariableop5
1savev2_res_conv_a_repr_a_bias_read_readvariableop8
4savev2_batch_normalization_gamma_read_readvariableop7
3savev2_batch_normalization_beta_read_readvariableop>
:savev2_batch_normalization_moving_mean_read_readvariableopB
>savev2_batch_normalization_moving_variance_read_readvariableop7
3savev2_res_conv_b_repr_a_kernel_read_readvariableop5
1savev2_res_conv_b_repr_a_bias_read_readvariableop:
6savev2_batch_normalization_1_gamma_read_readvariableop9
5savev2_batch_normalization_1_beta_read_readvariableop@
<savev2_batch_normalization_1_moving_mean_read_readvariableopD
@savev2_batch_normalization_1_moving_variance_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop7
3savev2_res_conv_a_repr_b_kernel_read_readvariableop5
1savev2_res_conv_a_repr_b_bias_read_readvariableop:
6savev2_batch_normalization_2_gamma_read_readvariableop9
5savev2_batch_normalization_2_beta_read_readvariableop@
<savev2_batch_normalization_2_moving_mean_read_readvariableopD
@savev2_batch_normalization_2_moving_variance_read_readvariableop7
3savev2_res_conv_b_repr_b_kernel_read_readvariableop5
1savev2_res_conv_b_repr_b_bias_read_readvariableop:
6savev2_batch_normalization_3_gamma_read_readvariableop9
5savev2_batch_normalization_3_beta_read_readvariableop@
<savev2_batch_normalization_3_moving_mean_read_readvariableopD
@savev2_batch_normalization_3_moving_variance_read_readvariableop7
3savev2_res_conv_a_repr_c_kernel_read_readvariableop5
1savev2_res_conv_a_repr_c_bias_read_readvariableop:
6savev2_batch_normalization_4_gamma_read_readvariableop9
5savev2_batch_normalization_4_beta_read_readvariableop@
<savev2_batch_normalization_4_moving_mean_read_readvariableopD
@savev2_batch_normalization_4_moving_variance_read_readvariableop7
3savev2_res_conv_b_repr_c_kernel_read_readvariableop5
1savev2_res_conv_b_repr_c_bias_read_readvariableop:
6savev2_batch_normalization_5_gamma_read_readvariableop9
5savev2_batch_normalization_5_beta_read_readvariableop@
<savev2_batch_normalization_5_moving_mean_read_readvariableopD
@savev2_batch_normalization_5_moving_variance_read_readvariableop7
3savev2_res_conv_a_repr_d_kernel_read_readvariableop5
1savev2_res_conv_a_repr_d_bias_read_readvariableop:
6savev2_batch_normalization_6_gamma_read_readvariableop9
5savev2_batch_normalization_6_beta_read_readvariableop@
<savev2_batch_normalization_6_moving_mean_read_readvariableopD
@savev2_batch_normalization_6_moving_variance_read_readvariableop7
3savev2_res_conv_b_repr_d_kernel_read_readvariableop5
1savev2_res_conv_b_repr_d_bias_read_readvariableop:
6savev2_batch_normalization_7_gamma_read_readvariableop9
5savev2_batch_normalization_7_beta_read_readvariableop@
<savev2_batch_normalization_7_moving_mean_read_readvariableopD
@savev2_batch_normalization_7_moving_variance_read_readvariableop0
,savev2_repr_board_kernel_read_readvariableop.
*savev2_repr_board_bias_read_readvariableop
savev2_1_const

identity_1��MergeV2Checkpoints�SaveV2�SaveV2_1�
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_5588681c84c440f0a4868fefa86232ad/part2
StringJoin/inputs_1�

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*�
value�B�6B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop3savev2_res_conv_a_repr_a_kernel_read_readvariableop1savev2_res_conv_a_repr_a_bias_read_readvariableop4savev2_batch_normalization_gamma_read_readvariableop3savev2_batch_normalization_beta_read_readvariableop:savev2_batch_normalization_moving_mean_read_readvariableop>savev2_batch_normalization_moving_variance_read_readvariableop3savev2_res_conv_b_repr_a_kernel_read_readvariableop1savev2_res_conv_b_repr_a_bias_read_readvariableop6savev2_batch_normalization_1_gamma_read_readvariableop5savev2_batch_normalization_1_beta_read_readvariableop<savev2_batch_normalization_1_moving_mean_read_readvariableop@savev2_batch_normalization_1_moving_variance_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop3savev2_res_conv_a_repr_b_kernel_read_readvariableop1savev2_res_conv_a_repr_b_bias_read_readvariableop6savev2_batch_normalization_2_gamma_read_readvariableop5savev2_batch_normalization_2_beta_read_readvariableop<savev2_batch_normalization_2_moving_mean_read_readvariableop@savev2_batch_normalization_2_moving_variance_read_readvariableop3savev2_res_conv_b_repr_b_kernel_read_readvariableop1savev2_res_conv_b_repr_b_bias_read_readvariableop6savev2_batch_normalization_3_gamma_read_readvariableop5savev2_batch_normalization_3_beta_read_readvariableop<savev2_batch_normalization_3_moving_mean_read_readvariableop@savev2_batch_normalization_3_moving_variance_read_readvariableop3savev2_res_conv_a_repr_c_kernel_read_readvariableop1savev2_res_conv_a_repr_c_bias_read_readvariableop6savev2_batch_normalization_4_gamma_read_readvariableop5savev2_batch_normalization_4_beta_read_readvariableop<savev2_batch_normalization_4_moving_mean_read_readvariableop@savev2_batch_normalization_4_moving_variance_read_readvariableop3savev2_res_conv_b_repr_c_kernel_read_readvariableop1savev2_res_conv_b_repr_c_bias_read_readvariableop6savev2_batch_normalization_5_gamma_read_readvariableop5savev2_batch_normalization_5_beta_read_readvariableop<savev2_batch_normalization_5_moving_mean_read_readvariableop@savev2_batch_normalization_5_moving_variance_read_readvariableop3savev2_res_conv_a_repr_d_kernel_read_readvariableop1savev2_res_conv_a_repr_d_bias_read_readvariableop6savev2_batch_normalization_6_gamma_read_readvariableop5savev2_batch_normalization_6_beta_read_readvariableop<savev2_batch_normalization_6_moving_mean_read_readvariableop@savev2_batch_normalization_6_moving_variance_read_readvariableop3savev2_res_conv_b_repr_d_kernel_read_readvariableop1savev2_res_conv_b_repr_d_bias_read_readvariableop6savev2_batch_normalization_7_gamma_read_readvariableop5savev2_batch_normalization_7_beta_read_readvariableop<savev2_batch_normalization_7_moving_mean_read_readvariableop@savev2_batch_normalization_7_moving_variance_read_readvariableop,savev2_repr_board_kernel_read_readvariableop*savev2_repr_board_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *D
dtypes:
8262
SaveV2�
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shard�
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1�
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_names�
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices�
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identity�

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*�
_input_shapes�
�: : : :  : : : : : :  : : : : : : @:@:@@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@:@@:@:@:@:@:@:@:: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
��
�#
I__inference_Representation_layer_call_and_return_conditional_losses_24360	
board)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_24
0res_conv_a_repr_a_statefulpartitionedcall_args_14
0res_conv_a_repr_a_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_16
2batch_normalization_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_36
2batch_normalization_statefulpartitionedcall_args_44
0res_conv_b_repr_a_statefulpartitionedcall_args_14
0res_conv_b_repr_a_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_18
4batch_normalization_1_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_38
4batch_normalization_1_statefulpartitionedcall_args_4+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_24
0res_conv_a_repr_b_statefulpartitionedcall_args_14
0res_conv_a_repr_b_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_18
4batch_normalization_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_38
4batch_normalization_2_statefulpartitionedcall_args_44
0res_conv_b_repr_b_statefulpartitionedcall_args_14
0res_conv_b_repr_b_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_18
4batch_normalization_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_38
4batch_normalization_3_statefulpartitionedcall_args_44
0res_conv_a_repr_c_statefulpartitionedcall_args_14
0res_conv_a_repr_c_statefulpartitionedcall_args_28
4batch_normalization_4_statefulpartitionedcall_args_18
4batch_normalization_4_statefulpartitionedcall_args_28
4batch_normalization_4_statefulpartitionedcall_args_38
4batch_normalization_4_statefulpartitionedcall_args_44
0res_conv_b_repr_c_statefulpartitionedcall_args_14
0res_conv_b_repr_c_statefulpartitionedcall_args_28
4batch_normalization_5_statefulpartitionedcall_args_18
4batch_normalization_5_statefulpartitionedcall_args_28
4batch_normalization_5_statefulpartitionedcall_args_38
4batch_normalization_5_statefulpartitionedcall_args_44
0res_conv_a_repr_d_statefulpartitionedcall_args_14
0res_conv_a_repr_d_statefulpartitionedcall_args_28
4batch_normalization_6_statefulpartitionedcall_args_18
4batch_normalization_6_statefulpartitionedcall_args_28
4batch_normalization_6_statefulpartitionedcall_args_38
4batch_normalization_6_statefulpartitionedcall_args_44
0res_conv_b_repr_d_statefulpartitionedcall_args_14
0res_conv_b_repr_d_statefulpartitionedcall_args_28
4batch_normalization_7_statefulpartitionedcall_args_18
4batch_normalization_7_statefulpartitionedcall_args_28
4batch_normalization_7_statefulpartitionedcall_args_38
4batch_normalization_7_statefulpartitionedcall_args_4-
)repr_board_statefulpartitionedcall_args_1-
)repr_board_statefulpartitionedcall_args_2
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�"repr_board/StatefulPartitionedCall�1repr_board/bias/Regularizer/Square/ReadVariableOp�3repr_board/kernel/Regularizer/Square/ReadVariableOp�)res_conv_A_repr_a/StatefulPartitionedCall�:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp�)res_conv_A_repr_b/StatefulPartitionedCall�:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp�)res_conv_A_repr_c/StatefulPartitionedCall�:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp�)res_conv_A_repr_d/StatefulPartitionedCall�:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp�)res_conv_B_repr_a/StatefulPartitionedCall�:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp�)res_conv_B_repr_b/StatefulPartitionedCall�:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp�)res_conv_B_repr_c/StatefulPartitionedCall�:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp�)res_conv_B_repr_d/StatefulPartitionedCall�:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp�
reshape/PartitionedCallPartitionedCallboard*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:���������``*/
config_proto

GPU

CPU2 *0J 8*K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_232432
reshape/PartitionedCall�
permute/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:���������``*/
config_proto

GPU

CPU2 *0J 8*K
fFRD
B__inference_permute_layer_call_and_return_conditional_losses_218372
permute/PartitionedCall�
reshape_1/PartitionedCallPartitionedCall permute/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������``*/
config_proto

GPU

CPU2 *0J 8*M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_232662
reshape_1/PartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_218552 
conv2d/StatefulPartitionedCall�
)res_conv_A_repr_a/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:00res_conv_a_repr_a_statefulpartitionedcall_args_10res_conv_a_repr_a_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_A_repr_a_layer_call_and_return_conditional_losses_218832+
)res_conv_A_repr_a/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall2res_conv_A_repr_a/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_232852
activation/PartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:02batch_normalization_statefulpartitionedcall_args_12batch_normalization_statefulpartitionedcall_args_22batch_normalization_statefulpartitionedcall_args_32batch_normalization_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_233482-
+batch_normalization/StatefulPartitionedCall�
)res_conv_B_repr_a/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:00res_conv_b_repr_a_statefulpartitionedcall_args_10res_conv_b_repr_a_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_B_repr_a_layer_call_and_return_conditional_losses_220432+
)res_conv_B_repr_a/StatefulPartitionedCall�
activation_1/PartitionedCallPartitionedCall2res_conv_B_repr_a/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_233802
activation_1/PartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:04batch_normalization_1_statefulpartitionedcall_args_14batch_normalization_1_statefulpartitionedcall_args_24batch_normalization_1_statefulpartitionedcall_args_34batch_normalization_1_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_234432/
-batch_normalization_1/StatefulPartitionedCall�
add/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_234732
add/PartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_221952"
 conv2d_1/StatefulPartitionedCall�
)res_conv_A_repr_b/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:00res_conv_a_repr_b_statefulpartitionedcall_args_10res_conv_a_repr_b_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_A_repr_b_layer_call_and_return_conditional_losses_222232+
)res_conv_A_repr_b/StatefulPartitionedCall�
activation_2/PartitionedCallPartitionedCall2res_conv_A_repr_b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_234932
activation_2/PartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:04batch_normalization_2_statefulpartitionedcall_args_14batch_normalization_2_statefulpartitionedcall_args_24batch_normalization_2_statefulpartitionedcall_args_34batch_normalization_2_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_235562/
-batch_normalization_2/StatefulPartitionedCall�
)res_conv_B_repr_b/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:00res_conv_b_repr_b_statefulpartitionedcall_args_10res_conv_b_repr_b_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_B_repr_b_layer_call_and_return_conditional_losses_223832+
)res_conv_B_repr_b/StatefulPartitionedCall�
activation_3/PartitionedCallPartitionedCall2res_conv_B_repr_b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_235882
activation_3/PartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:04batch_normalization_3_statefulpartitionedcall_args_14batch_normalization_3_statefulpartitionedcall_args_24batch_normalization_3_statefulpartitionedcall_args_34batch_normalization_3_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_236512/
-batch_normalization_3/StatefulPartitionedCall�
add_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_236812
add_1/PartitionedCall�
!average_pooling2d/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_225292#
!average_pooling2d/PartitionedCall�
)res_conv_A_repr_c/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:00res_conv_a_repr_c_statefulpartitionedcall_args_10res_conv_a_repr_c_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_A_repr_c_layer_call_and_return_conditional_losses_225552+
)res_conv_A_repr_c/StatefulPartitionedCall�
activation_4/PartitionedCallPartitionedCall2res_conv_A_repr_c/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_236992
activation_4/PartitionedCall�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:04batch_normalization_4_statefulpartitionedcall_args_14batch_normalization_4_statefulpartitionedcall_args_24batch_normalization_4_statefulpartitionedcall_args_34batch_normalization_4_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_237622/
-batch_normalization_4/StatefulPartitionedCall�
)res_conv_B_repr_c/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:00res_conv_b_repr_c_statefulpartitionedcall_args_10res_conv_b_repr_c_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_B_repr_c_layer_call_and_return_conditional_losses_227152+
)res_conv_B_repr_c/StatefulPartitionedCall�
activation_5/PartitionedCallPartitionedCall2res_conv_B_repr_c/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_237942
activation_5/PartitionedCall�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:04batch_normalization_5_statefulpartitionedcall_args_14batch_normalization_5_statefulpartitionedcall_args_24batch_normalization_5_statefulpartitionedcall_args_34batch_normalization_5_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_238572/
-batch_normalization_5/StatefulPartitionedCall�
add_2/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*I
fDRB
@__inference_add_2_layer_call_and_return_conditional_losses_238872
add_2/PartitionedCall�
#average_pooling2d_1/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*W
fRRP
N__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_228612%
#average_pooling2d_1/PartitionedCall�
)res_conv_A_repr_d/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_1/PartitionedCall:output:00res_conv_a_repr_d_statefulpartitionedcall_args_10res_conv_a_repr_d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_A_repr_d_layer_call_and_return_conditional_losses_228872+
)res_conv_A_repr_d/StatefulPartitionedCall�
activation_6/PartitionedCallPartitionedCall2res_conv_A_repr_d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_6_layer_call_and_return_conditional_losses_239052
activation_6/PartitionedCall�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:04batch_normalization_6_statefulpartitionedcall_args_14batch_normalization_6_statefulpartitionedcall_args_24batch_normalization_6_statefulpartitionedcall_args_34batch_normalization_6_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_239682/
-batch_normalization_6/StatefulPartitionedCall�
)res_conv_B_repr_d/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:00res_conv_b_repr_d_statefulpartitionedcall_args_10res_conv_b_repr_d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_B_repr_d_layer_call_and_return_conditional_losses_230472+
)res_conv_B_repr_d/StatefulPartitionedCall�
activation_7/PartitionedCallPartitionedCall2res_conv_B_repr_d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_7_layer_call_and_return_conditional_losses_240002
activation_7/PartitionedCall�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:04batch_normalization_7_statefulpartitionedcall_args_14batch_normalization_7_statefulpartitionedcall_args_24batch_normalization_7_statefulpartitionedcall_args_34batch_normalization_7_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_240632/
-batch_normalization_7/StatefulPartitionedCall�
add_3/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0,average_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*I
fDRB
@__inference_add_3_layer_call_and_return_conditional_losses_240932
add_3/PartitionedCall�
"repr_board/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0)repr_board_statefulpartitionedcall_args_1)repr_board_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������*/
config_proto

GPU

CPU2 *0J 8*N
fIRG
E__inference_repr_board_layer_call_and_return_conditional_losses_232162$
"repr_board/StatefulPartitionedCall�
:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_a_repr_a_statefulpartitionedcall_args_1*^res_conv_A_repr_a/StatefulPartitionedCall*&
_output_shapes
:  *
dtype02<
:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_a/kernel/Regularizer/SquareSquareBres_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2-
+res_conv_A_repr_a/kernel/Regularizer/Square�
*res_conv_A_repr_a/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_a/kernel/Regularizer/Const�
(res_conv_A_repr_a/kernel/Regularizer/SumSum/res_conv_A_repr_a/kernel/Regularizer/Square:y:03res_conv_A_repr_a/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_a/kernel/Regularizer/Sum�
*res_conv_A_repr_a/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_a/kernel/Regularizer/mul/x�
(res_conv_A_repr_a/kernel/Regularizer/mulMul3res_conv_A_repr_a/kernel/Regularizer/mul/x:output:01res_conv_A_repr_a/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_a/kernel/Regularizer/mul�
*res_conv_A_repr_a/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_a/kernel/Regularizer/add/x�
(res_conv_A_repr_a/kernel/Regularizer/addAddV23res_conv_A_repr_a/kernel/Regularizer/add/x:output:0,res_conv_A_repr_a/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_a/kernel/Regularizer/add�
:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_b_repr_a_statefulpartitionedcall_args_1*^res_conv_B_repr_a/StatefulPartitionedCall*&
_output_shapes
:  *
dtype02<
:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_a/kernel/Regularizer/SquareSquareBres_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2-
+res_conv_B_repr_a/kernel/Regularizer/Square�
*res_conv_B_repr_a/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_a/kernel/Regularizer/Const�
(res_conv_B_repr_a/kernel/Regularizer/SumSum/res_conv_B_repr_a/kernel/Regularizer/Square:y:03res_conv_B_repr_a/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_a/kernel/Regularizer/Sum�
*res_conv_B_repr_a/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_a/kernel/Regularizer/mul/x�
(res_conv_B_repr_a/kernel/Regularizer/mulMul3res_conv_B_repr_a/kernel/Regularizer/mul/x:output:01res_conv_B_repr_a/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_a/kernel/Regularizer/mul�
*res_conv_B_repr_a/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_a/kernel/Regularizer/add/x�
(res_conv_B_repr_a/kernel/Regularizer/addAddV23res_conv_B_repr_a/kernel/Regularizer/add/x:output:0,res_conv_B_repr_a/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_a/kernel/Regularizer/add�
:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_a_repr_b_statefulpartitionedcall_args_1*^res_conv_A_repr_b/StatefulPartitionedCall*&
_output_shapes
:@@*
dtype02<
:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_b/kernel/Regularizer/SquareSquareBres_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_A_repr_b/kernel/Regularizer/Square�
*res_conv_A_repr_b/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_b/kernel/Regularizer/Const�
(res_conv_A_repr_b/kernel/Regularizer/SumSum/res_conv_A_repr_b/kernel/Regularizer/Square:y:03res_conv_A_repr_b/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_b/kernel/Regularizer/Sum�
*res_conv_A_repr_b/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_b/kernel/Regularizer/mul/x�
(res_conv_A_repr_b/kernel/Regularizer/mulMul3res_conv_A_repr_b/kernel/Regularizer/mul/x:output:01res_conv_A_repr_b/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_b/kernel/Regularizer/mul�
*res_conv_A_repr_b/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_b/kernel/Regularizer/add/x�
(res_conv_A_repr_b/kernel/Regularizer/addAddV23res_conv_A_repr_b/kernel/Regularizer/add/x:output:0,res_conv_A_repr_b/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_b/kernel/Regularizer/add�
:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_b_repr_b_statefulpartitionedcall_args_1*^res_conv_B_repr_b/StatefulPartitionedCall*&
_output_shapes
:@@*
dtype02<
:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_b/kernel/Regularizer/SquareSquareBres_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_B_repr_b/kernel/Regularizer/Square�
*res_conv_B_repr_b/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_b/kernel/Regularizer/Const�
(res_conv_B_repr_b/kernel/Regularizer/SumSum/res_conv_B_repr_b/kernel/Regularizer/Square:y:03res_conv_B_repr_b/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_b/kernel/Regularizer/Sum�
*res_conv_B_repr_b/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_b/kernel/Regularizer/mul/x�
(res_conv_B_repr_b/kernel/Regularizer/mulMul3res_conv_B_repr_b/kernel/Regularizer/mul/x:output:01res_conv_B_repr_b/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_b/kernel/Regularizer/mul�
*res_conv_B_repr_b/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_b/kernel/Regularizer/add/x�
(res_conv_B_repr_b/kernel/Regularizer/addAddV23res_conv_B_repr_b/kernel/Regularizer/add/x:output:0,res_conv_B_repr_b/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_b/kernel/Regularizer/add�
:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_a_repr_c_statefulpartitionedcall_args_1*^res_conv_A_repr_c/StatefulPartitionedCall*&
_output_shapes
:@@*
dtype02<
:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_c/kernel/Regularizer/SquareSquareBres_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_A_repr_c/kernel/Regularizer/Square�
*res_conv_A_repr_c/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_c/kernel/Regularizer/Const�
(res_conv_A_repr_c/kernel/Regularizer/SumSum/res_conv_A_repr_c/kernel/Regularizer/Square:y:03res_conv_A_repr_c/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_c/kernel/Regularizer/Sum�
*res_conv_A_repr_c/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_c/kernel/Regularizer/mul/x�
(res_conv_A_repr_c/kernel/Regularizer/mulMul3res_conv_A_repr_c/kernel/Regularizer/mul/x:output:01res_conv_A_repr_c/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_c/kernel/Regularizer/mul�
*res_conv_A_repr_c/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_c/kernel/Regularizer/add/x�
(res_conv_A_repr_c/kernel/Regularizer/addAddV23res_conv_A_repr_c/kernel/Regularizer/add/x:output:0,res_conv_A_repr_c/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_c/kernel/Regularizer/add�
:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_b_repr_c_statefulpartitionedcall_args_1*^res_conv_B_repr_c/StatefulPartitionedCall*&
_output_shapes
:@@*
dtype02<
:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_c/kernel/Regularizer/SquareSquareBres_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_B_repr_c/kernel/Regularizer/Square�
*res_conv_B_repr_c/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_c/kernel/Regularizer/Const�
(res_conv_B_repr_c/kernel/Regularizer/SumSum/res_conv_B_repr_c/kernel/Regularizer/Square:y:03res_conv_B_repr_c/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_c/kernel/Regularizer/Sum�
*res_conv_B_repr_c/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_c/kernel/Regularizer/mul/x�
(res_conv_B_repr_c/kernel/Regularizer/mulMul3res_conv_B_repr_c/kernel/Regularizer/mul/x:output:01res_conv_B_repr_c/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_c/kernel/Regularizer/mul�
*res_conv_B_repr_c/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_c/kernel/Regularizer/add/x�
(res_conv_B_repr_c/kernel/Regularizer/addAddV23res_conv_B_repr_c/kernel/Regularizer/add/x:output:0,res_conv_B_repr_c/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_c/kernel/Regularizer/add�
:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_a_repr_d_statefulpartitionedcall_args_1*^res_conv_A_repr_d/StatefulPartitionedCall*&
_output_shapes
:@@*
dtype02<
:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_d/kernel/Regularizer/SquareSquareBres_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_A_repr_d/kernel/Regularizer/Square�
*res_conv_A_repr_d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_d/kernel/Regularizer/Const�
(res_conv_A_repr_d/kernel/Regularizer/SumSum/res_conv_A_repr_d/kernel/Regularizer/Square:y:03res_conv_A_repr_d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_d/kernel/Regularizer/Sum�
*res_conv_A_repr_d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_d/kernel/Regularizer/mul/x�
(res_conv_A_repr_d/kernel/Regularizer/mulMul3res_conv_A_repr_d/kernel/Regularizer/mul/x:output:01res_conv_A_repr_d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_d/kernel/Regularizer/mul�
*res_conv_A_repr_d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_d/kernel/Regularizer/add/x�
(res_conv_A_repr_d/kernel/Regularizer/addAddV23res_conv_A_repr_d/kernel/Regularizer/add/x:output:0,res_conv_A_repr_d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_d/kernel/Regularizer/add�
:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_b_repr_d_statefulpartitionedcall_args_1*^res_conv_B_repr_d/StatefulPartitionedCall*&
_output_shapes
:@@*
dtype02<
:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_d/kernel/Regularizer/SquareSquareBres_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_B_repr_d/kernel/Regularizer/Square�
*res_conv_B_repr_d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_d/kernel/Regularizer/Const�
(res_conv_B_repr_d/kernel/Regularizer/SumSum/res_conv_B_repr_d/kernel/Regularizer/Square:y:03res_conv_B_repr_d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_d/kernel/Regularizer/Sum�
*res_conv_B_repr_d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_d/kernel/Regularizer/mul/x�
(res_conv_B_repr_d/kernel/Regularizer/mulMul3res_conv_B_repr_d/kernel/Regularizer/mul/x:output:01res_conv_B_repr_d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_d/kernel/Regularizer/mul�
*res_conv_B_repr_d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_d/kernel/Regularizer/add/x�
(res_conv_B_repr_d/kernel/Regularizer/addAddV23res_conv_B_repr_d/kernel/Regularizer/add/x:output:0,res_conv_B_repr_d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_d/kernel/Regularizer/add�
3repr_board/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)repr_board_statefulpartitionedcall_args_1#^repr_board/StatefulPartitionedCall*&
_output_shapes
:@*
dtype025
3repr_board/kernel/Regularizer/Square/ReadVariableOp�
$repr_board/kernel/Regularizer/SquareSquare;repr_board/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2&
$repr_board/kernel/Regularizer/Square�
#repr_board/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2%
#repr_board/kernel/Regularizer/Const�
!repr_board/kernel/Regularizer/SumSum(repr_board/kernel/Regularizer/Square:y:0,repr_board/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!repr_board/kernel/Regularizer/Sum�
#repr_board/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72%
#repr_board/kernel/Regularizer/mul/x�
!repr_board/kernel/Regularizer/mulMul,repr_board/kernel/Regularizer/mul/x:output:0*repr_board/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!repr_board/kernel/Regularizer/mul�
#repr_board/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#repr_board/kernel/Regularizer/add/x�
!repr_board/kernel/Regularizer/addAddV2,repr_board/kernel/Regularizer/add/x:output:0%repr_board/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2#
!repr_board/kernel/Regularizer/add�
1repr_board/bias/Regularizer/Square/ReadVariableOpReadVariableOp)repr_board_statefulpartitionedcall_args_2#^repr_board/StatefulPartitionedCall*
_output_shapes
:*
dtype023
1repr_board/bias/Regularizer/Square/ReadVariableOp�
"repr_board/bias/Regularizer/SquareSquare9repr_board/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2$
"repr_board/bias/Regularizer/Square�
!repr_board/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!repr_board/bias/Regularizer/Const�
repr_board/bias/Regularizer/SumSum&repr_board/bias/Regularizer/Square:y:0*repr_board/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
repr_board/bias/Regularizer/Sum�
!repr_board/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72#
!repr_board/bias/Regularizer/mul/x�
repr_board/bias/Regularizer/mulMul*repr_board/bias/Regularizer/mul/x:output:0(repr_board/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
repr_board/bias/Regularizer/mul�
!repr_board/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!repr_board/bias/Regularizer/add/x�
repr_board/bias/Regularizer/addAddV2*repr_board/bias/Regularizer/add/x:output:0#repr_board/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
repr_board/bias/Regularizer/add�
IdentityIdentity+repr_board/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall#^repr_board/StatefulPartitionedCall2^repr_board/bias/Regularizer/Square/ReadVariableOp4^repr_board/kernel/Regularizer/Square/ReadVariableOp*^res_conv_A_repr_a/StatefulPartitionedCall;^res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp*^res_conv_A_repr_b/StatefulPartitionedCall;^res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp*^res_conv_A_repr_c/StatefulPartitionedCall;^res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp*^res_conv_A_repr_d/StatefulPartitionedCall;^res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp*^res_conv_B_repr_a/StatefulPartitionedCall;^res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp*^res_conv_B_repr_b/StatefulPartitionedCall;^res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp*^res_conv_B_repr_c/StatefulPartitionedCall;^res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp*^res_conv_B_repr_d/StatefulPartitionedCall;^res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������``::::::::::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2H
"repr_board/StatefulPartitionedCall"repr_board/StatefulPartitionedCall2f
1repr_board/bias/Regularizer/Square/ReadVariableOp1repr_board/bias/Regularizer/Square/ReadVariableOp2j
3repr_board/kernel/Regularizer/Square/ReadVariableOp3repr_board/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_A_repr_a/StatefulPartitionedCall)res_conv_A_repr_a/StatefulPartitionedCall2x
:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_A_repr_b/StatefulPartitionedCall)res_conv_A_repr_b/StatefulPartitionedCall2x
:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_A_repr_c/StatefulPartitionedCall)res_conv_A_repr_c/StatefulPartitionedCall2x
:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_A_repr_d/StatefulPartitionedCall)res_conv_A_repr_d/StatefulPartitionedCall2x
:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_B_repr_a/StatefulPartitionedCall)res_conv_B_repr_a/StatefulPartitionedCall2x
:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_B_repr_b/StatefulPartitionedCall)res_conv_B_repr_b/StatefulPartitionedCall2x
:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_B_repr_c/StatefulPartitionedCall)res_conv_B_repr_c/StatefulPartitionedCall2x
:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_B_repr_d/StatefulPartitionedCall)res_conv_B_repr_d/StatefulPartitionedCall2x
:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp:% !

_user_specified_nameboard
��
�0
I__inference_Representation_layer_call_and_return_conditional_losses_25823

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource4
0res_conv_a_repr_a_conv2d_readvariableop_resource5
1res_conv_a_repr_a_biasadd_readvariableop_resource/
+batch_normalization_readvariableop_resource1
-batch_normalization_readvariableop_1_resource@
<batch_normalization_fusedbatchnormv3_readvariableop_resourceB
>batch_normalization_fusedbatchnormv3_readvariableop_1_resource4
0res_conv_b_repr_a_conv2d_readvariableop_resource5
1res_conv_b_repr_a_biasadd_readvariableop_resource1
-batch_normalization_1_readvariableop_resource3
/batch_normalization_1_readvariableop_1_resourceB
>batch_normalization_1_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource4
0res_conv_a_repr_b_conv2d_readvariableop_resource5
1res_conv_a_repr_b_biasadd_readvariableop_resource1
-batch_normalization_2_readvariableop_resource3
/batch_normalization_2_readvariableop_1_resourceB
>batch_normalization_2_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource4
0res_conv_b_repr_b_conv2d_readvariableop_resource5
1res_conv_b_repr_b_biasadd_readvariableop_resource1
-batch_normalization_3_readvariableop_resource3
/batch_normalization_3_readvariableop_1_resourceB
>batch_normalization_3_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource4
0res_conv_a_repr_c_conv2d_readvariableop_resource5
1res_conv_a_repr_c_biasadd_readvariableop_resource1
-batch_normalization_4_readvariableop_resource3
/batch_normalization_4_readvariableop_1_resourceB
>batch_normalization_4_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource4
0res_conv_b_repr_c_conv2d_readvariableop_resource5
1res_conv_b_repr_c_biasadd_readvariableop_resource1
-batch_normalization_5_readvariableop_resource3
/batch_normalization_5_readvariableop_1_resourceB
>batch_normalization_5_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource4
0res_conv_a_repr_d_conv2d_readvariableop_resource5
1res_conv_a_repr_d_biasadd_readvariableop_resource1
-batch_normalization_6_readvariableop_resource3
/batch_normalization_6_readvariableop_1_resourceB
>batch_normalization_6_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource4
0res_conv_b_repr_d_conv2d_readvariableop_resource5
1res_conv_b_repr_d_biasadd_readvariableop_resource1
-batch_normalization_7_readvariableop_resource3
/batch_normalization_7_readvariableop_1_resourceB
>batch_normalization_7_fusedbatchnormv3_readvariableop_resourceD
@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource-
)repr_board_conv2d_readvariableop_resource.
*repr_board_biasadd_readvariableop_resource
identity��3batch_normalization/FusedBatchNormV3/ReadVariableOp�5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�"batch_normalization/ReadVariableOp�$batch_normalization/ReadVariableOp_1�5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_1/ReadVariableOp�&batch_normalization_1/ReadVariableOp_1�5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_2/ReadVariableOp�&batch_normalization_2/ReadVariableOp_1�5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_3/ReadVariableOp�&batch_normalization_3/ReadVariableOp_1�5batch_normalization_4/FusedBatchNormV3/ReadVariableOp�7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_4/ReadVariableOp�&batch_normalization_4/ReadVariableOp_1�5batch_normalization_5/FusedBatchNormV3/ReadVariableOp�7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_5/ReadVariableOp�&batch_normalization_5/ReadVariableOp_1�5batch_normalization_6/FusedBatchNormV3/ReadVariableOp�7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_6/ReadVariableOp�&batch_normalization_6/ReadVariableOp_1�5batch_normalization_7/FusedBatchNormV3/ReadVariableOp�7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1�$batch_normalization_7/ReadVariableOp�&batch_normalization_7/ReadVariableOp_1�conv2d/BiasAdd/ReadVariableOp�conv2d/Conv2D/ReadVariableOp�conv2d_1/BiasAdd/ReadVariableOp�conv2d_1/Conv2D/ReadVariableOp�!repr_board/BiasAdd/ReadVariableOp� repr_board/Conv2D/ReadVariableOp�1repr_board/bias/Regularizer/Square/ReadVariableOp�3repr_board/kernel/Regularizer/Square/ReadVariableOp�(res_conv_A_repr_a/BiasAdd/ReadVariableOp�'res_conv_A_repr_a/Conv2D/ReadVariableOp�:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp�(res_conv_A_repr_b/BiasAdd/ReadVariableOp�'res_conv_A_repr_b/Conv2D/ReadVariableOp�:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp�(res_conv_A_repr_c/BiasAdd/ReadVariableOp�'res_conv_A_repr_c/Conv2D/ReadVariableOp�:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp�(res_conv_A_repr_d/BiasAdd/ReadVariableOp�'res_conv_A_repr_d/Conv2D/ReadVariableOp�:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp�(res_conv_B_repr_a/BiasAdd/ReadVariableOp�'res_conv_B_repr_a/Conv2D/ReadVariableOp�:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp�(res_conv_B_repr_b/BiasAdd/ReadVariableOp�'res_conv_B_repr_b/Conv2D/ReadVariableOp�:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp�(res_conv_B_repr_c/BiasAdd/ReadVariableOp�'res_conv_B_repr_c/Conv2D/ReadVariableOp�:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp�(res_conv_B_repr_d/BiasAdd/ReadVariableOp�'res_conv_B_repr_d/Conv2D/ReadVariableOp�:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOpT
reshape/ShapeShapeinputs*
T0*
_output_shapes
:2
reshape/Shape�
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack�
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1�
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2�
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :`2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :`2
reshape/Reshape/shape/3t
reshape/Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/4�
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0 reshape/Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape�
reshape/ReshapeReshapeinputsreshape/Reshape/shape:output:0*
T0*3
_output_shapes!
:���������``2
reshape/Reshape�
permute/transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
permute/transpose/perm�
permute/transpose	Transposereshape/Reshape:output:0permute/transpose/perm:output:0*
T0*3
_output_shapes!
:���������``2
permute/transposeg
reshape_1/ShapeShapepermute/transpose:y:0*
T0*
_output_shapes
:2
reshape_1/Shape�
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack�
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1�
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2�
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :`2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :`2
reshape_1/Reshape/shape/2x
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/3�
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape�
reshape_1/ReshapeReshapepermute/transpose:y:0 reshape_1/Reshape/shape:output:0*
T0*/
_output_shapes
:���������``2
reshape_1/Reshape�
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp�
conv2d/Conv2DConv2Dreshape_1/Reshape:output:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00 *
paddingSAME*
strides
2
conv2d/Conv2D�
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp�
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00 2
conv2d/BiasAdd�
'res_conv_A_repr_a/Conv2D/ReadVariableOpReadVariableOp0res_conv_a_repr_a_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'res_conv_A_repr_a/Conv2D/ReadVariableOp�
res_conv_A_repr_a/Conv2DConv2Dconv2d/BiasAdd:output:0/res_conv_A_repr_a/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00 *
paddingSAME*
strides
2
res_conv_A_repr_a/Conv2D�
(res_conv_A_repr_a/BiasAdd/ReadVariableOpReadVariableOp1res_conv_a_repr_a_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(res_conv_A_repr_a/BiasAdd/ReadVariableOp�
res_conv_A_repr_a/BiasAddBiasAdd!res_conv_A_repr_a/Conv2D:output:00res_conv_A_repr_a/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00 2
res_conv_A_repr_a/BiasAdd�
activation/ReluRelu"res_conv_A_repr_a/BiasAdd:output:0*
T0*/
_output_shapes
:���������00 2
activation/Relu�
 batch_normalization/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2"
 batch_normalization/LogicalAnd/x�
 batch_normalization/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2"
 batch_normalization/LogicalAnd/y�
batch_normalization/LogicalAnd
LogicalAnd)batch_normalization/LogicalAnd/x:output:0)batch_normalization/LogicalAnd/y:output:0*
_output_shapes
: 2 
batch_normalization/LogicalAnd�
"batch_normalization/ReadVariableOpReadVariableOp+batch_normalization_readvariableop_resource*
_output_shapes
: *
dtype02$
"batch_normalization/ReadVariableOp�
$batch_normalization/ReadVariableOp_1ReadVariableOp-batch_normalization_readvariableop_1_resource*
_output_shapes
: *
dtype02&
$batch_normalization/ReadVariableOp_1�
3batch_normalization/FusedBatchNormV3/ReadVariableOpReadVariableOp<batch_normalization_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype025
3batch_normalization/FusedBatchNormV3/ReadVariableOp�
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp>batch_normalization_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype027
5batch_normalization/FusedBatchNormV3/ReadVariableOp_1�
$batch_normalization/FusedBatchNormV3FusedBatchNormV3activation/Relu:activations:0*batch_normalization/ReadVariableOp:value:0,batch_normalization/ReadVariableOp_1:value:0;batch_normalization/FusedBatchNormV3/ReadVariableOp:value:0=batch_normalization/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������00 : : : : :*
epsilon%o�:*
is_training( 2&
$batch_normalization/FusedBatchNormV3{
batch_normalization/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization/Const�
'res_conv_B_repr_a/Conv2D/ReadVariableOpReadVariableOp0res_conv_b_repr_a_conv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02)
'res_conv_B_repr_a/Conv2D/ReadVariableOp�
res_conv_B_repr_a/Conv2DConv2D(batch_normalization/FusedBatchNormV3:y:0/res_conv_B_repr_a/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00 *
paddingSAME*
strides
2
res_conv_B_repr_a/Conv2D�
(res_conv_B_repr_a/BiasAdd/ReadVariableOpReadVariableOp1res_conv_b_repr_a_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(res_conv_B_repr_a/BiasAdd/ReadVariableOp�
res_conv_B_repr_a/BiasAddBiasAdd!res_conv_B_repr_a/Conv2D:output:00res_conv_B_repr_a/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������00 2
res_conv_B_repr_a/BiasAdd�
activation_1/ReluRelu"res_conv_B_repr_a/BiasAdd:output:0*
T0*/
_output_shapes
:���������00 2
activation_1/Relu�
"batch_normalization_1/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_1/LogicalAnd/x�
"batch_normalization_1/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_1/LogicalAnd/y�
 batch_normalization_1/LogicalAnd
LogicalAnd+batch_normalization_1/LogicalAnd/x:output:0+batch_normalization_1/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_1/LogicalAnd�
$batch_normalization_1/ReadVariableOpReadVariableOp-batch_normalization_1_readvariableop_resource*
_output_shapes
: *
dtype02&
$batch_normalization_1/ReadVariableOp�
&batch_normalization_1/ReadVariableOp_1ReadVariableOp/batch_normalization_1_readvariableop_1_resource*
_output_shapes
: *
dtype02(
&batch_normalization_1/ReadVariableOp_1�
5batch_normalization_1/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_1_fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype027
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_1_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype029
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_1/FusedBatchNormV3FusedBatchNormV3activation_1/Relu:activations:0,batch_normalization_1/ReadVariableOp:value:0.batch_normalization_1/ReadVariableOp_1:value:0=batch_normalization_1/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������00 : : : : :*
epsilon%o�:*
is_training( 2(
&batch_normalization_1/FusedBatchNormV3
batch_normalization_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_1/Const�
add/addAddV2*batch_normalization_1/FusedBatchNormV3:y:0conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:���������00 2	
add/add�
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp�
conv2d_1/Conv2DConv2Dadd/add:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
conv2d_1/Conv2D�
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp�
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
conv2d_1/BiasAdd�
'res_conv_A_repr_b/Conv2D/ReadVariableOpReadVariableOp0res_conv_a_repr_b_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'res_conv_A_repr_b/Conv2D/ReadVariableOp�
res_conv_A_repr_b/Conv2DConv2Dconv2d_1/BiasAdd:output:0/res_conv_A_repr_b/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
res_conv_A_repr_b/Conv2D�
(res_conv_A_repr_b/BiasAdd/ReadVariableOpReadVariableOp1res_conv_a_repr_b_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(res_conv_A_repr_b/BiasAdd/ReadVariableOp�
res_conv_A_repr_b/BiasAddBiasAdd!res_conv_A_repr_b/Conv2D:output:00res_conv_A_repr_b/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
res_conv_A_repr_b/BiasAdd�
activation_2/ReluRelu"res_conv_A_repr_b/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
activation_2/Relu�
"batch_normalization_2/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_2/LogicalAnd/x�
"batch_normalization_2/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_2/LogicalAnd/y�
 batch_normalization_2/LogicalAnd
LogicalAnd+batch_normalization_2/LogicalAnd/x:output:0+batch_normalization_2/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_2/LogicalAnd�
$batch_normalization_2/ReadVariableOpReadVariableOp-batch_normalization_2_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_2/ReadVariableOp�
&batch_normalization_2/ReadVariableOp_1ReadVariableOp/batch_normalization_2_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_2/ReadVariableOp_1�
5batch_normalization_2/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_2_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_2_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_2/FusedBatchNormV3FusedBatchNormV3activation_2/Relu:activations:0,batch_normalization_2/ReadVariableOp:value:0.batch_normalization_2/ReadVariableOp_1:value:0=batch_normalization_2/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2(
&batch_normalization_2/FusedBatchNormV3
batch_normalization_2/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_2/Const�
'res_conv_B_repr_b/Conv2D/ReadVariableOpReadVariableOp0res_conv_b_repr_b_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'res_conv_B_repr_b/Conv2D/ReadVariableOp�
res_conv_B_repr_b/Conv2DConv2D*batch_normalization_2/FusedBatchNormV3:y:0/res_conv_B_repr_b/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
res_conv_B_repr_b/Conv2D�
(res_conv_B_repr_b/BiasAdd/ReadVariableOpReadVariableOp1res_conv_b_repr_b_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(res_conv_B_repr_b/BiasAdd/ReadVariableOp�
res_conv_B_repr_b/BiasAddBiasAdd!res_conv_B_repr_b/Conv2D:output:00res_conv_B_repr_b/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
res_conv_B_repr_b/BiasAdd�
activation_3/ReluRelu"res_conv_B_repr_b/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
activation_3/Relu�
"batch_normalization_3/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_3/LogicalAnd/x�
"batch_normalization_3/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_3/LogicalAnd/y�
 batch_normalization_3/LogicalAnd
LogicalAnd+batch_normalization_3/LogicalAnd/x:output:0+batch_normalization_3/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_3/LogicalAnd�
$batch_normalization_3/ReadVariableOpReadVariableOp-batch_normalization_3_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_3/ReadVariableOp�
&batch_normalization_3/ReadVariableOp_1ReadVariableOp/batch_normalization_3_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_3/ReadVariableOp_1�
5batch_normalization_3/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_3_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_3_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_3/FusedBatchNormV3FusedBatchNormV3activation_3/Relu:activations:0,batch_normalization_3/ReadVariableOp:value:0.batch_normalization_3/ReadVariableOp_1:value:0=batch_normalization_3/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2(
&batch_normalization_3/FusedBatchNormV3
batch_normalization_3/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_3/Const�
	add_1/addAddV2*batch_normalization_3/FusedBatchNormV3:y:0conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
	add_1/add�
average_pooling2d/AvgPoolAvgPooladd_1/add:z:0*
T0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
average_pooling2d/AvgPool�
'res_conv_A_repr_c/Conv2D/ReadVariableOpReadVariableOp0res_conv_a_repr_c_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'res_conv_A_repr_c/Conv2D/ReadVariableOp�
res_conv_A_repr_c/Conv2DConv2D"average_pooling2d/AvgPool:output:0/res_conv_A_repr_c/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
res_conv_A_repr_c/Conv2D�
(res_conv_A_repr_c/BiasAdd/ReadVariableOpReadVariableOp1res_conv_a_repr_c_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(res_conv_A_repr_c/BiasAdd/ReadVariableOp�
res_conv_A_repr_c/BiasAddBiasAdd!res_conv_A_repr_c/Conv2D:output:00res_conv_A_repr_c/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
res_conv_A_repr_c/BiasAdd�
activation_4/ReluRelu"res_conv_A_repr_c/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
activation_4/Relu�
"batch_normalization_4/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_4/LogicalAnd/x�
"batch_normalization_4/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_4/LogicalAnd/y�
 batch_normalization_4/LogicalAnd
LogicalAnd+batch_normalization_4/LogicalAnd/x:output:0+batch_normalization_4/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_4/LogicalAnd�
$batch_normalization_4/ReadVariableOpReadVariableOp-batch_normalization_4_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_4/ReadVariableOp�
&batch_normalization_4/ReadVariableOp_1ReadVariableOp/batch_normalization_4_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_4/ReadVariableOp_1�
5batch_normalization_4/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_4_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_4_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_4/FusedBatchNormV3FusedBatchNormV3activation_4/Relu:activations:0,batch_normalization_4/ReadVariableOp:value:0.batch_normalization_4/ReadVariableOp_1:value:0=batch_normalization_4/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2(
&batch_normalization_4/FusedBatchNormV3
batch_normalization_4/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_4/Const�
'res_conv_B_repr_c/Conv2D/ReadVariableOpReadVariableOp0res_conv_b_repr_c_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'res_conv_B_repr_c/Conv2D/ReadVariableOp�
res_conv_B_repr_c/Conv2DConv2D*batch_normalization_4/FusedBatchNormV3:y:0/res_conv_B_repr_c/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
res_conv_B_repr_c/Conv2D�
(res_conv_B_repr_c/BiasAdd/ReadVariableOpReadVariableOp1res_conv_b_repr_c_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(res_conv_B_repr_c/BiasAdd/ReadVariableOp�
res_conv_B_repr_c/BiasAddBiasAdd!res_conv_B_repr_c/Conv2D:output:00res_conv_B_repr_c/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
res_conv_B_repr_c/BiasAdd�
activation_5/ReluRelu"res_conv_B_repr_c/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
activation_5/Relu�
"batch_normalization_5/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_5/LogicalAnd/x�
"batch_normalization_5/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_5/LogicalAnd/y�
 batch_normalization_5/LogicalAnd
LogicalAnd+batch_normalization_5/LogicalAnd/x:output:0+batch_normalization_5/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_5/LogicalAnd�
$batch_normalization_5/ReadVariableOpReadVariableOp-batch_normalization_5_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_5/ReadVariableOp�
&batch_normalization_5/ReadVariableOp_1ReadVariableOp/batch_normalization_5_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_5/ReadVariableOp_1�
5batch_normalization_5/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_5_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_5_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_5/FusedBatchNormV3FusedBatchNormV3activation_5/Relu:activations:0,batch_normalization_5/ReadVariableOp:value:0.batch_normalization_5/ReadVariableOp_1:value:0=batch_normalization_5/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2(
&batch_normalization_5/FusedBatchNormV3
batch_normalization_5/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_5/Const�
	add_2/addAddV2*batch_normalization_5/FusedBatchNormV3:y:0"average_pooling2d/AvgPool:output:0*
T0*/
_output_shapes
:���������@2
	add_2/add�
average_pooling2d_1/AvgPoolAvgPooladd_2/add:z:0*
T0*/
_output_shapes
:���������@*
ksize
*
paddingVALID*
strides
2
average_pooling2d_1/AvgPool�
'res_conv_A_repr_d/Conv2D/ReadVariableOpReadVariableOp0res_conv_a_repr_d_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'res_conv_A_repr_d/Conv2D/ReadVariableOp�
res_conv_A_repr_d/Conv2DConv2D$average_pooling2d_1/AvgPool:output:0/res_conv_A_repr_d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
res_conv_A_repr_d/Conv2D�
(res_conv_A_repr_d/BiasAdd/ReadVariableOpReadVariableOp1res_conv_a_repr_d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(res_conv_A_repr_d/BiasAdd/ReadVariableOp�
res_conv_A_repr_d/BiasAddBiasAdd!res_conv_A_repr_d/Conv2D:output:00res_conv_A_repr_d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
res_conv_A_repr_d/BiasAdd�
activation_6/ReluRelu"res_conv_A_repr_d/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
activation_6/Relu�
"batch_normalization_6/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_6/LogicalAnd/x�
"batch_normalization_6/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_6/LogicalAnd/y�
 batch_normalization_6/LogicalAnd
LogicalAnd+batch_normalization_6/LogicalAnd/x:output:0+batch_normalization_6/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_6/LogicalAnd�
$batch_normalization_6/ReadVariableOpReadVariableOp-batch_normalization_6_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_6/ReadVariableOp�
&batch_normalization_6/ReadVariableOp_1ReadVariableOp/batch_normalization_6_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_6/ReadVariableOp_1�
5batch_normalization_6/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_6_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_6_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_6/FusedBatchNormV3FusedBatchNormV3activation_6/Relu:activations:0,batch_normalization_6/ReadVariableOp:value:0.batch_normalization_6/ReadVariableOp_1:value:0=batch_normalization_6/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2(
&batch_normalization_6/FusedBatchNormV3
batch_normalization_6/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_6/Const�
'res_conv_B_repr_d/Conv2D/ReadVariableOpReadVariableOp0res_conv_b_repr_d_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'res_conv_B_repr_d/Conv2D/ReadVariableOp�
res_conv_B_repr_d/Conv2DConv2D*batch_normalization_6/FusedBatchNormV3:y:0/res_conv_B_repr_d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@*
paddingSAME*
strides
2
res_conv_B_repr_d/Conv2D�
(res_conv_B_repr_d/BiasAdd/ReadVariableOpReadVariableOp1res_conv_b_repr_d_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(res_conv_B_repr_d/BiasAdd/ReadVariableOp�
res_conv_B_repr_d/BiasAddBiasAdd!res_conv_B_repr_d/Conv2D:output:00res_conv_B_repr_d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@2
res_conv_B_repr_d/BiasAdd�
activation_7/ReluRelu"res_conv_B_repr_d/BiasAdd:output:0*
T0*/
_output_shapes
:���������@2
activation_7/Relu�
"batch_normalization_7/LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2$
"batch_normalization_7/LogicalAnd/x�
"batch_normalization_7/LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2$
"batch_normalization_7/LogicalAnd/y�
 batch_normalization_7/LogicalAnd
LogicalAnd+batch_normalization_7/LogicalAnd/x:output:0+batch_normalization_7/LogicalAnd/y:output:0*
_output_shapes
: 2"
 batch_normalization_7/LogicalAnd�
$batch_normalization_7/ReadVariableOpReadVariableOp-batch_normalization_7_readvariableop_resource*
_output_shapes
:@*
dtype02&
$batch_normalization_7/ReadVariableOp�
&batch_normalization_7/ReadVariableOp_1ReadVariableOp/batch_normalization_7_readvariableop_1_resource*
_output_shapes
:@*
dtype02(
&batch_normalization_7/ReadVariableOp_1�
5batch_normalization_7/FusedBatchNormV3/ReadVariableOpReadVariableOp>batch_normalization_7_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype027
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp�
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1ReadVariableOp@batch_normalization_7_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype029
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1�
&batch_normalization_7/FusedBatchNormV3FusedBatchNormV3activation_7/Relu:activations:0,batch_normalization_7/ReadVariableOp:value:0.batch_normalization_7/ReadVariableOp_1:value:0=batch_normalization_7/FusedBatchNormV3/ReadVariableOp:value:0?batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2(
&batch_normalization_7/FusedBatchNormV3
batch_normalization_7/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
batch_normalization_7/Const�
	add_3/addAddV2*batch_normalization_7/FusedBatchNormV3:y:0$average_pooling2d_1/AvgPool:output:0*
T0*/
_output_shapes
:���������@2
	add_3/add�
 repr_board/Conv2D/ReadVariableOpReadVariableOp)repr_board_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype02"
 repr_board/Conv2D/ReadVariableOp�
repr_board/Conv2DConv2Dadd_3/add:z:0(repr_board/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������*
paddingSAME*
strides
2
repr_board/Conv2D�
!repr_board/BiasAdd/ReadVariableOpReadVariableOp*repr_board_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02#
!repr_board/BiasAdd/ReadVariableOp�
repr_board/BiasAddBiasAddrepr_board/Conv2D:output:0)repr_board/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������2
repr_board/BiasAdd�
repr_board/ReluRelurepr_board/BiasAdd:output:0*
T0*/
_output_shapes
:���������2
repr_board/Relu�
:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_a_repr_a_conv2d_readvariableop_resource(^res_conv_A_repr_a/Conv2D/ReadVariableOp*&
_output_shapes
:  *
dtype02<
:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_a/kernel/Regularizer/SquareSquareBres_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2-
+res_conv_A_repr_a/kernel/Regularizer/Square�
*res_conv_A_repr_a/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_a/kernel/Regularizer/Const�
(res_conv_A_repr_a/kernel/Regularizer/SumSum/res_conv_A_repr_a/kernel/Regularizer/Square:y:03res_conv_A_repr_a/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_a/kernel/Regularizer/Sum�
*res_conv_A_repr_a/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_a/kernel/Regularizer/mul/x�
(res_conv_A_repr_a/kernel/Regularizer/mulMul3res_conv_A_repr_a/kernel/Regularizer/mul/x:output:01res_conv_A_repr_a/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_a/kernel/Regularizer/mul�
*res_conv_A_repr_a/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_a/kernel/Regularizer/add/x�
(res_conv_A_repr_a/kernel/Regularizer/addAddV23res_conv_A_repr_a/kernel/Regularizer/add/x:output:0,res_conv_A_repr_a/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_a/kernel/Regularizer/add�
:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_b_repr_a_conv2d_readvariableop_resource(^res_conv_B_repr_a/Conv2D/ReadVariableOp*&
_output_shapes
:  *
dtype02<
:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_a/kernel/Regularizer/SquareSquareBres_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2-
+res_conv_B_repr_a/kernel/Regularizer/Square�
*res_conv_B_repr_a/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_a/kernel/Regularizer/Const�
(res_conv_B_repr_a/kernel/Regularizer/SumSum/res_conv_B_repr_a/kernel/Regularizer/Square:y:03res_conv_B_repr_a/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_a/kernel/Regularizer/Sum�
*res_conv_B_repr_a/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_a/kernel/Regularizer/mul/x�
(res_conv_B_repr_a/kernel/Regularizer/mulMul3res_conv_B_repr_a/kernel/Regularizer/mul/x:output:01res_conv_B_repr_a/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_a/kernel/Regularizer/mul�
*res_conv_B_repr_a/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_a/kernel/Regularizer/add/x�
(res_conv_B_repr_a/kernel/Regularizer/addAddV23res_conv_B_repr_a/kernel/Regularizer/add/x:output:0,res_conv_B_repr_a/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_a/kernel/Regularizer/add�
:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_a_repr_b_conv2d_readvariableop_resource(^res_conv_A_repr_b/Conv2D/ReadVariableOp*&
_output_shapes
:@@*
dtype02<
:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_b/kernel/Regularizer/SquareSquareBres_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_A_repr_b/kernel/Regularizer/Square�
*res_conv_A_repr_b/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_b/kernel/Regularizer/Const�
(res_conv_A_repr_b/kernel/Regularizer/SumSum/res_conv_A_repr_b/kernel/Regularizer/Square:y:03res_conv_A_repr_b/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_b/kernel/Regularizer/Sum�
*res_conv_A_repr_b/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_b/kernel/Regularizer/mul/x�
(res_conv_A_repr_b/kernel/Regularizer/mulMul3res_conv_A_repr_b/kernel/Regularizer/mul/x:output:01res_conv_A_repr_b/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_b/kernel/Regularizer/mul�
*res_conv_A_repr_b/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_b/kernel/Regularizer/add/x�
(res_conv_A_repr_b/kernel/Regularizer/addAddV23res_conv_A_repr_b/kernel/Regularizer/add/x:output:0,res_conv_A_repr_b/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_b/kernel/Regularizer/add�
:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_b_repr_b_conv2d_readvariableop_resource(^res_conv_B_repr_b/Conv2D/ReadVariableOp*&
_output_shapes
:@@*
dtype02<
:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_b/kernel/Regularizer/SquareSquareBres_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_B_repr_b/kernel/Regularizer/Square�
*res_conv_B_repr_b/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_b/kernel/Regularizer/Const�
(res_conv_B_repr_b/kernel/Regularizer/SumSum/res_conv_B_repr_b/kernel/Regularizer/Square:y:03res_conv_B_repr_b/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_b/kernel/Regularizer/Sum�
*res_conv_B_repr_b/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_b/kernel/Regularizer/mul/x�
(res_conv_B_repr_b/kernel/Regularizer/mulMul3res_conv_B_repr_b/kernel/Regularizer/mul/x:output:01res_conv_B_repr_b/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_b/kernel/Regularizer/mul�
*res_conv_B_repr_b/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_b/kernel/Regularizer/add/x�
(res_conv_B_repr_b/kernel/Regularizer/addAddV23res_conv_B_repr_b/kernel/Regularizer/add/x:output:0,res_conv_B_repr_b/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_b/kernel/Regularizer/add�
:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_a_repr_c_conv2d_readvariableop_resource(^res_conv_A_repr_c/Conv2D/ReadVariableOp*&
_output_shapes
:@@*
dtype02<
:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_c/kernel/Regularizer/SquareSquareBres_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_A_repr_c/kernel/Regularizer/Square�
*res_conv_A_repr_c/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_c/kernel/Regularizer/Const�
(res_conv_A_repr_c/kernel/Regularizer/SumSum/res_conv_A_repr_c/kernel/Regularizer/Square:y:03res_conv_A_repr_c/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_c/kernel/Regularizer/Sum�
*res_conv_A_repr_c/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_c/kernel/Regularizer/mul/x�
(res_conv_A_repr_c/kernel/Regularizer/mulMul3res_conv_A_repr_c/kernel/Regularizer/mul/x:output:01res_conv_A_repr_c/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_c/kernel/Regularizer/mul�
*res_conv_A_repr_c/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_c/kernel/Regularizer/add/x�
(res_conv_A_repr_c/kernel/Regularizer/addAddV23res_conv_A_repr_c/kernel/Regularizer/add/x:output:0,res_conv_A_repr_c/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_c/kernel/Regularizer/add�
:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_b_repr_c_conv2d_readvariableop_resource(^res_conv_B_repr_c/Conv2D/ReadVariableOp*&
_output_shapes
:@@*
dtype02<
:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_c/kernel/Regularizer/SquareSquareBres_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_B_repr_c/kernel/Regularizer/Square�
*res_conv_B_repr_c/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_c/kernel/Regularizer/Const�
(res_conv_B_repr_c/kernel/Regularizer/SumSum/res_conv_B_repr_c/kernel/Regularizer/Square:y:03res_conv_B_repr_c/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_c/kernel/Regularizer/Sum�
*res_conv_B_repr_c/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_c/kernel/Regularizer/mul/x�
(res_conv_B_repr_c/kernel/Regularizer/mulMul3res_conv_B_repr_c/kernel/Regularizer/mul/x:output:01res_conv_B_repr_c/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_c/kernel/Regularizer/mul�
*res_conv_B_repr_c/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_c/kernel/Regularizer/add/x�
(res_conv_B_repr_c/kernel/Regularizer/addAddV23res_conv_B_repr_c/kernel/Regularizer/add/x:output:0,res_conv_B_repr_c/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_c/kernel/Regularizer/add�
:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_a_repr_d_conv2d_readvariableop_resource(^res_conv_A_repr_d/Conv2D/ReadVariableOp*&
_output_shapes
:@@*
dtype02<
:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_d/kernel/Regularizer/SquareSquareBres_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_A_repr_d/kernel/Regularizer/Square�
*res_conv_A_repr_d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_d/kernel/Regularizer/Const�
(res_conv_A_repr_d/kernel/Regularizer/SumSum/res_conv_A_repr_d/kernel/Regularizer/Square:y:03res_conv_A_repr_d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_d/kernel/Regularizer/Sum�
*res_conv_A_repr_d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_d/kernel/Regularizer/mul/x�
(res_conv_A_repr_d/kernel/Regularizer/mulMul3res_conv_A_repr_d/kernel/Regularizer/mul/x:output:01res_conv_A_repr_d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_d/kernel/Regularizer/mul�
*res_conv_A_repr_d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_d/kernel/Regularizer/add/x�
(res_conv_A_repr_d/kernel/Regularizer/addAddV23res_conv_A_repr_d/kernel/Regularizer/add/x:output:0,res_conv_A_repr_d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_d/kernel/Regularizer/add�
:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_b_repr_d_conv2d_readvariableop_resource(^res_conv_B_repr_d/Conv2D/ReadVariableOp*&
_output_shapes
:@@*
dtype02<
:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_d/kernel/Regularizer/SquareSquareBres_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_B_repr_d/kernel/Regularizer/Square�
*res_conv_B_repr_d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_d/kernel/Regularizer/Const�
(res_conv_B_repr_d/kernel/Regularizer/SumSum/res_conv_B_repr_d/kernel/Regularizer/Square:y:03res_conv_B_repr_d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_d/kernel/Regularizer/Sum�
*res_conv_B_repr_d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_d/kernel/Regularizer/mul/x�
(res_conv_B_repr_d/kernel/Regularizer/mulMul3res_conv_B_repr_d/kernel/Regularizer/mul/x:output:01res_conv_B_repr_d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_d/kernel/Regularizer/mul�
*res_conv_B_repr_d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_d/kernel/Regularizer/add/x�
(res_conv_B_repr_d/kernel/Regularizer/addAddV23res_conv_B_repr_d/kernel/Regularizer/add/x:output:0,res_conv_B_repr_d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_d/kernel/Regularizer/add�
3repr_board/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)repr_board_conv2d_readvariableop_resource!^repr_board/Conv2D/ReadVariableOp*&
_output_shapes
:@*
dtype025
3repr_board/kernel/Regularizer/Square/ReadVariableOp�
$repr_board/kernel/Regularizer/SquareSquare;repr_board/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2&
$repr_board/kernel/Regularizer/Square�
#repr_board/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2%
#repr_board/kernel/Regularizer/Const�
!repr_board/kernel/Regularizer/SumSum(repr_board/kernel/Regularizer/Square:y:0,repr_board/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!repr_board/kernel/Regularizer/Sum�
#repr_board/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72%
#repr_board/kernel/Regularizer/mul/x�
!repr_board/kernel/Regularizer/mulMul,repr_board/kernel/Regularizer/mul/x:output:0*repr_board/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!repr_board/kernel/Regularizer/mul�
#repr_board/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#repr_board/kernel/Regularizer/add/x�
!repr_board/kernel/Regularizer/addAddV2,repr_board/kernel/Regularizer/add/x:output:0%repr_board/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2#
!repr_board/kernel/Regularizer/add�
1repr_board/bias/Regularizer/Square/ReadVariableOpReadVariableOp*repr_board_biasadd_readvariableop_resource"^repr_board/BiasAdd/ReadVariableOp*
_output_shapes
:*
dtype023
1repr_board/bias/Regularizer/Square/ReadVariableOp�
"repr_board/bias/Regularizer/SquareSquare9repr_board/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2$
"repr_board/bias/Regularizer/Square�
!repr_board/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!repr_board/bias/Regularizer/Const�
repr_board/bias/Regularizer/SumSum&repr_board/bias/Regularizer/Square:y:0*repr_board/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
repr_board/bias/Regularizer/Sum�
!repr_board/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72#
!repr_board/bias/Regularizer/mul/x�
repr_board/bias/Regularizer/mulMul*repr_board/bias/Regularizer/mul/x:output:0(repr_board/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
repr_board/bias/Regularizer/mul�
!repr_board/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!repr_board/bias/Regularizer/add/x�
repr_board/bias/Regularizer/addAddV2*repr_board/bias/Regularizer/add/x:output:0#repr_board/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
repr_board/bias/Regularizer/add�
IdentityIdentityrepr_board/Relu:activations:04^batch_normalization/FusedBatchNormV3/ReadVariableOp6^batch_normalization/FusedBatchNormV3/ReadVariableOp_1#^batch_normalization/ReadVariableOp%^batch_normalization/ReadVariableOp_16^batch_normalization_1/FusedBatchNormV3/ReadVariableOp8^batch_normalization_1/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_1/ReadVariableOp'^batch_normalization_1/ReadVariableOp_16^batch_normalization_2/FusedBatchNormV3/ReadVariableOp8^batch_normalization_2/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_2/ReadVariableOp'^batch_normalization_2/ReadVariableOp_16^batch_normalization_3/FusedBatchNormV3/ReadVariableOp8^batch_normalization_3/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_3/ReadVariableOp'^batch_normalization_3/ReadVariableOp_16^batch_normalization_4/FusedBatchNormV3/ReadVariableOp8^batch_normalization_4/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_4/ReadVariableOp'^batch_normalization_4/ReadVariableOp_16^batch_normalization_5/FusedBatchNormV3/ReadVariableOp8^batch_normalization_5/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_5/ReadVariableOp'^batch_normalization_5/ReadVariableOp_16^batch_normalization_6/FusedBatchNormV3/ReadVariableOp8^batch_normalization_6/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_6/ReadVariableOp'^batch_normalization_6/ReadVariableOp_16^batch_normalization_7/FusedBatchNormV3/ReadVariableOp8^batch_normalization_7/FusedBatchNormV3/ReadVariableOp_1%^batch_normalization_7/ReadVariableOp'^batch_normalization_7/ReadVariableOp_1^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp"^repr_board/BiasAdd/ReadVariableOp!^repr_board/Conv2D/ReadVariableOp2^repr_board/bias/Regularizer/Square/ReadVariableOp4^repr_board/kernel/Regularizer/Square/ReadVariableOp)^res_conv_A_repr_a/BiasAdd/ReadVariableOp(^res_conv_A_repr_a/Conv2D/ReadVariableOp;^res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp)^res_conv_A_repr_b/BiasAdd/ReadVariableOp(^res_conv_A_repr_b/Conv2D/ReadVariableOp;^res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp)^res_conv_A_repr_c/BiasAdd/ReadVariableOp(^res_conv_A_repr_c/Conv2D/ReadVariableOp;^res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp)^res_conv_A_repr_d/BiasAdd/ReadVariableOp(^res_conv_A_repr_d/Conv2D/ReadVariableOp;^res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp)^res_conv_B_repr_a/BiasAdd/ReadVariableOp(^res_conv_B_repr_a/Conv2D/ReadVariableOp;^res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp)^res_conv_B_repr_b/BiasAdd/ReadVariableOp(^res_conv_B_repr_b/Conv2D/ReadVariableOp;^res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp)^res_conv_B_repr_c/BiasAdd/ReadVariableOp(^res_conv_B_repr_c/Conv2D/ReadVariableOp;^res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp)^res_conv_B_repr_d/BiasAdd/ReadVariableOp(^res_conv_B_repr_d/Conv2D/ReadVariableOp;^res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������``::::::::::::::::::::::::::::::::::::::::::::::::::::::2j
3batch_normalization/FusedBatchNormV3/ReadVariableOp3batch_normalization/FusedBatchNormV3/ReadVariableOp2n
5batch_normalization/FusedBatchNormV3/ReadVariableOp_15batch_normalization/FusedBatchNormV3/ReadVariableOp_12H
"batch_normalization/ReadVariableOp"batch_normalization/ReadVariableOp2L
$batch_normalization/ReadVariableOp_1$batch_normalization/ReadVariableOp_12n
5batch_normalization_1/FusedBatchNormV3/ReadVariableOp5batch_normalization_1/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_1/FusedBatchNormV3/ReadVariableOp_17batch_normalization_1/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_1/ReadVariableOp$batch_normalization_1/ReadVariableOp2P
&batch_normalization_1/ReadVariableOp_1&batch_normalization_1/ReadVariableOp_12n
5batch_normalization_2/FusedBatchNormV3/ReadVariableOp5batch_normalization_2/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_2/FusedBatchNormV3/ReadVariableOp_17batch_normalization_2/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_2/ReadVariableOp$batch_normalization_2/ReadVariableOp2P
&batch_normalization_2/ReadVariableOp_1&batch_normalization_2/ReadVariableOp_12n
5batch_normalization_3/FusedBatchNormV3/ReadVariableOp5batch_normalization_3/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_3/FusedBatchNormV3/ReadVariableOp_17batch_normalization_3/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_3/ReadVariableOp$batch_normalization_3/ReadVariableOp2P
&batch_normalization_3/ReadVariableOp_1&batch_normalization_3/ReadVariableOp_12n
5batch_normalization_4/FusedBatchNormV3/ReadVariableOp5batch_normalization_4/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_4/FusedBatchNormV3/ReadVariableOp_17batch_normalization_4/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_4/ReadVariableOp$batch_normalization_4/ReadVariableOp2P
&batch_normalization_4/ReadVariableOp_1&batch_normalization_4/ReadVariableOp_12n
5batch_normalization_5/FusedBatchNormV3/ReadVariableOp5batch_normalization_5/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_5/FusedBatchNormV3/ReadVariableOp_17batch_normalization_5/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_5/ReadVariableOp$batch_normalization_5/ReadVariableOp2P
&batch_normalization_5/ReadVariableOp_1&batch_normalization_5/ReadVariableOp_12n
5batch_normalization_6/FusedBatchNormV3/ReadVariableOp5batch_normalization_6/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_6/FusedBatchNormV3/ReadVariableOp_17batch_normalization_6/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_6/ReadVariableOp$batch_normalization_6/ReadVariableOp2P
&batch_normalization_6/ReadVariableOp_1&batch_normalization_6/ReadVariableOp_12n
5batch_normalization_7/FusedBatchNormV3/ReadVariableOp5batch_normalization_7/FusedBatchNormV3/ReadVariableOp2r
7batch_normalization_7/FusedBatchNormV3/ReadVariableOp_17batch_normalization_7/FusedBatchNormV3/ReadVariableOp_12L
$batch_normalization_7/ReadVariableOp$batch_normalization_7/ReadVariableOp2P
&batch_normalization_7/ReadVariableOp_1&batch_normalization_7/ReadVariableOp_12>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2F
!repr_board/BiasAdd/ReadVariableOp!repr_board/BiasAdd/ReadVariableOp2D
 repr_board/Conv2D/ReadVariableOp repr_board/Conv2D/ReadVariableOp2f
1repr_board/bias/Regularizer/Square/ReadVariableOp1repr_board/bias/Regularizer/Square/ReadVariableOp2j
3repr_board/kernel/Regularizer/Square/ReadVariableOp3repr_board/kernel/Regularizer/Square/ReadVariableOp2T
(res_conv_A_repr_a/BiasAdd/ReadVariableOp(res_conv_A_repr_a/BiasAdd/ReadVariableOp2R
'res_conv_A_repr_a/Conv2D/ReadVariableOp'res_conv_A_repr_a/Conv2D/ReadVariableOp2x
:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp2T
(res_conv_A_repr_b/BiasAdd/ReadVariableOp(res_conv_A_repr_b/BiasAdd/ReadVariableOp2R
'res_conv_A_repr_b/Conv2D/ReadVariableOp'res_conv_A_repr_b/Conv2D/ReadVariableOp2x
:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp2T
(res_conv_A_repr_c/BiasAdd/ReadVariableOp(res_conv_A_repr_c/BiasAdd/ReadVariableOp2R
'res_conv_A_repr_c/Conv2D/ReadVariableOp'res_conv_A_repr_c/Conv2D/ReadVariableOp2x
:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp2T
(res_conv_A_repr_d/BiasAdd/ReadVariableOp(res_conv_A_repr_d/BiasAdd/ReadVariableOp2R
'res_conv_A_repr_d/Conv2D/ReadVariableOp'res_conv_A_repr_d/Conv2D/ReadVariableOp2x
:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp2T
(res_conv_B_repr_a/BiasAdd/ReadVariableOp(res_conv_B_repr_a/BiasAdd/ReadVariableOp2R
'res_conv_B_repr_a/Conv2D/ReadVariableOp'res_conv_B_repr_a/Conv2D/ReadVariableOp2x
:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp2T
(res_conv_B_repr_b/BiasAdd/ReadVariableOp(res_conv_B_repr_b/BiasAdd/ReadVariableOp2R
'res_conv_B_repr_b/Conv2D/ReadVariableOp'res_conv_B_repr_b/Conv2D/ReadVariableOp2x
:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp2T
(res_conv_B_repr_c/BiasAdd/ReadVariableOp(res_conv_B_repr_c/BiasAdd/ReadVariableOp2R
'res_conv_B_repr_c/Conv2D/ReadVariableOp'res_conv_B_repr_c/Conv2D/ReadVariableOp2x
:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp2T
(res_conv_B_repr_d/BiasAdd/ReadVariableOp(res_conv_B_repr_d/BiasAdd/ReadVariableOp2R
'res_conv_B_repr_d/Conv2D/ReadVariableOp'res_conv_B_repr_d/Conv2D/ReadVariableOp2x
:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26434

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
3__inference_batch_normalization_layer_call_fn_26149

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+��������������������������� */
config_proto

GPU

CPU2 *0J 8*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_219852
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_3_layer_call_fn_26695

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_224852
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
*__inference_repr_board_layer_call_fn_23224

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������*/
config_proto

GPU

CPU2 *0J 8*N
fIRG
E__inference_repr_board_layer_call_and_return_conditional_losses_232162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
E
)__inference_reshape_1_layer_call_fn_25980

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������``*/
config_proto

GPU

CPU2 *0J 8*M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_232662
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������``2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������``:& "
 
_user_specified_nameinputs
�
h
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_22529

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
AvgPool�
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
l
@__inference_add_1_layer_call_and_return_conditional_losses_26710
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������@:���������@:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�
c
G__inference_activation_7_layer_call_and_return_conditional_losses_27275

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�#
�
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26854

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_26839
assignmovingavg_1_26846
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/26839*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/26839*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_26839*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/26839*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/26839*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_26839AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/26839*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/26846*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/26846*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_26846*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/26846*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/26846*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_26846AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/26846*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22176

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
L__inference_res_conv_B_repr_b_layer_call_and_return_conditional_losses_22383

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAdd�
:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource^Conv2D/ReadVariableOp*&
_output_shapes
:@@*
dtype02<
:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_b/kernel/Regularizer/SquareSquareBres_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_B_repr_b/kernel/Regularizer/Square�
*res_conv_B_repr_b/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_b/kernel/Regularizer/Const�
(res_conv_B_repr_b/kernel/Regularizer/SumSum/res_conv_B_repr_b/kernel/Regularizer/Square:y:03res_conv_B_repr_b/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_b/kernel/Regularizer/Sum�
*res_conv_B_repr_b/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_b/kernel/Regularizer/mul/x�
(res_conv_B_repr_b/kernel/Regularizer/mulMul3res_conv_B_repr_b/kernel/Regularizer/mul/x:output:01res_conv_B_repr_b/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_b/kernel/Regularizer/mul�
*res_conv_B_repr_b/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_b/kernel/Regularizer/add/x�
(res_conv_B_repr_b/kernel/Regularizer/addAddV23res_conv_B_repr_b/kernel/Regularizer/add/x:output:0,res_conv_B_repr_b/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_b/kernel/Regularizer/add�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp;^res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2x
:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
__inference_loss_fn_7_27572G
Cres_conv_b_repr_d_kernel_regularizer_square_readvariableop_resource
identity��:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp�
:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpCres_conv_b_repr_d_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:@@*
dtype02<
:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_d/kernel/Regularizer/SquareSquareBres_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_B_repr_d/kernel/Regularizer/Square�
*res_conv_B_repr_d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_d/kernel/Regularizer/Const�
(res_conv_B_repr_d/kernel/Regularizer/SumSum/res_conv_B_repr_d/kernel/Regularizer/Square:y:03res_conv_B_repr_d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_d/kernel/Regularizer/Sum�
*res_conv_B_repr_d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_d/kernel/Regularizer/mul/x�
(res_conv_B_repr_d/kernel/Regularizer/mulMul3res_conv_B_repr_d/kernel/Regularizer/mul/x:output:01res_conv_B_repr_d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_d/kernel/Regularizer/mul�
*res_conv_B_repr_d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_d/kernel/Regularizer/add/x�
(res_conv_B_repr_d/kernel/Regularizer/addAddV23res_conv_B_repr_d/kernel/Regularizer/add/x:output:0,res_conv_B_repr_d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_d/kernel/Regularizer/add�
IdentityIdentity,res_conv_B_repr_d/kernel/Regularizer/add:z:0;^res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2x
:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp
�
l
@__inference_add_3_layer_call_and_return_conditional_losses_27446
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������@:���������@:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
��
�#
I__inference_Representation_layer_call_and_return_conditional_losses_24186	
board)
%conv2d_statefulpartitionedcall_args_1)
%conv2d_statefulpartitionedcall_args_24
0res_conv_a_repr_a_statefulpartitionedcall_args_14
0res_conv_a_repr_a_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_16
2batch_normalization_statefulpartitionedcall_args_26
2batch_normalization_statefulpartitionedcall_args_36
2batch_normalization_statefulpartitionedcall_args_44
0res_conv_b_repr_a_statefulpartitionedcall_args_14
0res_conv_b_repr_a_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_18
4batch_normalization_1_statefulpartitionedcall_args_28
4batch_normalization_1_statefulpartitionedcall_args_38
4batch_normalization_1_statefulpartitionedcall_args_4+
'conv2d_1_statefulpartitionedcall_args_1+
'conv2d_1_statefulpartitionedcall_args_24
0res_conv_a_repr_b_statefulpartitionedcall_args_14
0res_conv_a_repr_b_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_18
4batch_normalization_2_statefulpartitionedcall_args_28
4batch_normalization_2_statefulpartitionedcall_args_38
4batch_normalization_2_statefulpartitionedcall_args_44
0res_conv_b_repr_b_statefulpartitionedcall_args_14
0res_conv_b_repr_b_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_18
4batch_normalization_3_statefulpartitionedcall_args_28
4batch_normalization_3_statefulpartitionedcall_args_38
4batch_normalization_3_statefulpartitionedcall_args_44
0res_conv_a_repr_c_statefulpartitionedcall_args_14
0res_conv_a_repr_c_statefulpartitionedcall_args_28
4batch_normalization_4_statefulpartitionedcall_args_18
4batch_normalization_4_statefulpartitionedcall_args_28
4batch_normalization_4_statefulpartitionedcall_args_38
4batch_normalization_4_statefulpartitionedcall_args_44
0res_conv_b_repr_c_statefulpartitionedcall_args_14
0res_conv_b_repr_c_statefulpartitionedcall_args_28
4batch_normalization_5_statefulpartitionedcall_args_18
4batch_normalization_5_statefulpartitionedcall_args_28
4batch_normalization_5_statefulpartitionedcall_args_38
4batch_normalization_5_statefulpartitionedcall_args_44
0res_conv_a_repr_d_statefulpartitionedcall_args_14
0res_conv_a_repr_d_statefulpartitionedcall_args_28
4batch_normalization_6_statefulpartitionedcall_args_18
4batch_normalization_6_statefulpartitionedcall_args_28
4batch_normalization_6_statefulpartitionedcall_args_38
4batch_normalization_6_statefulpartitionedcall_args_44
0res_conv_b_repr_d_statefulpartitionedcall_args_14
0res_conv_b_repr_d_statefulpartitionedcall_args_28
4batch_normalization_7_statefulpartitionedcall_args_18
4batch_normalization_7_statefulpartitionedcall_args_28
4batch_normalization_7_statefulpartitionedcall_args_38
4batch_normalization_7_statefulpartitionedcall_args_4-
)repr_board_statefulpartitionedcall_args_1-
)repr_board_statefulpartitionedcall_args_2
identity��+batch_normalization/StatefulPartitionedCall�-batch_normalization_1/StatefulPartitionedCall�-batch_normalization_2/StatefulPartitionedCall�-batch_normalization_3/StatefulPartitionedCall�-batch_normalization_4/StatefulPartitionedCall�-batch_normalization_5/StatefulPartitionedCall�-batch_normalization_6/StatefulPartitionedCall�-batch_normalization_7/StatefulPartitionedCall�conv2d/StatefulPartitionedCall� conv2d_1/StatefulPartitionedCall�"repr_board/StatefulPartitionedCall�1repr_board/bias/Regularizer/Square/ReadVariableOp�3repr_board/kernel/Regularizer/Square/ReadVariableOp�)res_conv_A_repr_a/StatefulPartitionedCall�:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp�)res_conv_A_repr_b/StatefulPartitionedCall�:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp�)res_conv_A_repr_c/StatefulPartitionedCall�:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp�)res_conv_A_repr_d/StatefulPartitionedCall�:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp�)res_conv_B_repr_a/StatefulPartitionedCall�:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp�)res_conv_B_repr_b/StatefulPartitionedCall�:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp�)res_conv_B_repr_c/StatefulPartitionedCall�:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp�)res_conv_B_repr_d/StatefulPartitionedCall�:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp�
reshape/PartitionedCallPartitionedCallboard*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:���������``*/
config_proto

GPU

CPU2 *0J 8*K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_232432
reshape/PartitionedCall�
permute/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*3
_output_shapes!
:���������``*/
config_proto

GPU

CPU2 *0J 8*K
fFRD
B__inference_permute_layer_call_and_return_conditional_losses_218372
permute/PartitionedCall�
reshape_1/PartitionedCallPartitionedCall permute/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������``*/
config_proto

GPU

CPU2 *0J 8*M
fHRF
D__inference_reshape_1_layer_call_and_return_conditional_losses_232662
reshape_1/PartitionedCall�
conv2d/StatefulPartitionedCallStatefulPartitionedCall"reshape_1/PartitionedCall:output:0%conv2d_statefulpartitionedcall_args_1%conv2d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*J
fERC
A__inference_conv2d_layer_call_and_return_conditional_losses_218552 
conv2d/StatefulPartitionedCall�
)res_conv_A_repr_a/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:00res_conv_a_repr_a_statefulpartitionedcall_args_10res_conv_a_repr_a_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_A_repr_a_layer_call_and_return_conditional_losses_218832+
)res_conv_A_repr_a/StatefulPartitionedCall�
activation/PartitionedCallPartitionedCall2res_conv_A_repr_a/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*N
fIRG
E__inference_activation_layer_call_and_return_conditional_losses_232852
activation/PartitionedCall�
+batch_normalization/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:02batch_normalization_statefulpartitionedcall_args_12batch_normalization_statefulpartitionedcall_args_22batch_normalization_statefulpartitionedcall_args_32batch_normalization_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_233262-
+batch_normalization/StatefulPartitionedCall�
)res_conv_B_repr_a/StatefulPartitionedCallStatefulPartitionedCall4batch_normalization/StatefulPartitionedCall:output:00res_conv_b_repr_a_statefulpartitionedcall_args_10res_conv_b_repr_a_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_B_repr_a_layer_call_and_return_conditional_losses_220432+
)res_conv_B_repr_a/StatefulPartitionedCall�
activation_1/PartitionedCallPartitionedCall2res_conv_B_repr_a/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_1_layer_call_and_return_conditional_losses_233802
activation_1/PartitionedCall�
-batch_normalization_1/StatefulPartitionedCallStatefulPartitionedCall%activation_1/PartitionedCall:output:04batch_normalization_1_statefulpartitionedcall_args_14batch_normalization_1_statefulpartitionedcall_args_24batch_normalization_1_statefulpartitionedcall_args_34batch_normalization_1_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_234212/
-batch_normalization_1/StatefulPartitionedCall�
add/PartitionedCallPartitionedCall6batch_normalization_1/StatefulPartitionedCall:output:0'conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������00 */
config_proto

GPU

CPU2 *0J 8*G
fBR@
>__inference_add_layer_call_and_return_conditional_losses_234732
add/PartitionedCall�
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0'conv2d_1_statefulpartitionedcall_args_1'conv2d_1_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_221952"
 conv2d_1/StatefulPartitionedCall�
)res_conv_A_repr_b/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:00res_conv_a_repr_b_statefulpartitionedcall_args_10res_conv_a_repr_b_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_A_repr_b_layer_call_and_return_conditional_losses_222232+
)res_conv_A_repr_b/StatefulPartitionedCall�
activation_2/PartitionedCallPartitionedCall2res_conv_A_repr_b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_2_layer_call_and_return_conditional_losses_234932
activation_2/PartitionedCall�
-batch_normalization_2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:04batch_normalization_2_statefulpartitionedcall_args_14batch_normalization_2_statefulpartitionedcall_args_24batch_normalization_2_statefulpartitionedcall_args_34batch_normalization_2_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_235342/
-batch_normalization_2/StatefulPartitionedCall�
)res_conv_B_repr_b/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_2/StatefulPartitionedCall:output:00res_conv_b_repr_b_statefulpartitionedcall_args_10res_conv_b_repr_b_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_B_repr_b_layer_call_and_return_conditional_losses_223832+
)res_conv_B_repr_b/StatefulPartitionedCall�
activation_3/PartitionedCallPartitionedCall2res_conv_B_repr_b/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_3_layer_call_and_return_conditional_losses_235882
activation_3/PartitionedCall�
-batch_normalization_3/StatefulPartitionedCallStatefulPartitionedCall%activation_3/PartitionedCall:output:04batch_normalization_3_statefulpartitionedcall_args_14batch_normalization_3_statefulpartitionedcall_args_24batch_normalization_3_statefulpartitionedcall_args_34batch_normalization_3_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_236292/
-batch_normalization_3/StatefulPartitionedCall�
add_1/PartitionedCallPartitionedCall6batch_normalization_3/StatefulPartitionedCall:output:0)conv2d_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*I
fDRB
@__inference_add_1_layer_call_and_return_conditional_losses_236812
add_1/PartitionedCall�
!average_pooling2d/PartitionedCallPartitionedCalladd_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_225292#
!average_pooling2d/PartitionedCall�
)res_conv_A_repr_c/StatefulPartitionedCallStatefulPartitionedCall*average_pooling2d/PartitionedCall:output:00res_conv_a_repr_c_statefulpartitionedcall_args_10res_conv_a_repr_c_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_A_repr_c_layer_call_and_return_conditional_losses_225552+
)res_conv_A_repr_c/StatefulPartitionedCall�
activation_4/PartitionedCallPartitionedCall2res_conv_A_repr_c/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_4_layer_call_and_return_conditional_losses_236992
activation_4/PartitionedCall�
-batch_normalization_4/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:04batch_normalization_4_statefulpartitionedcall_args_14batch_normalization_4_statefulpartitionedcall_args_24batch_normalization_4_statefulpartitionedcall_args_34batch_normalization_4_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_237402/
-batch_normalization_4/StatefulPartitionedCall�
)res_conv_B_repr_c/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_4/StatefulPartitionedCall:output:00res_conv_b_repr_c_statefulpartitionedcall_args_10res_conv_b_repr_c_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_B_repr_c_layer_call_and_return_conditional_losses_227152+
)res_conv_B_repr_c/StatefulPartitionedCall�
activation_5/PartitionedCallPartitionedCall2res_conv_B_repr_c/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_5_layer_call_and_return_conditional_losses_237942
activation_5/PartitionedCall�
-batch_normalization_5/StatefulPartitionedCallStatefulPartitionedCall%activation_5/PartitionedCall:output:04batch_normalization_5_statefulpartitionedcall_args_14batch_normalization_5_statefulpartitionedcall_args_24batch_normalization_5_statefulpartitionedcall_args_34batch_normalization_5_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_238352/
-batch_normalization_5/StatefulPartitionedCall�
add_2/PartitionedCallPartitionedCall6batch_normalization_5/StatefulPartitionedCall:output:0*average_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*I
fDRB
@__inference_add_2_layer_call_and_return_conditional_losses_238872
add_2/PartitionedCall�
#average_pooling2d_1/PartitionedCallPartitionedCalladd_2/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*W
fRRP
N__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_228612%
#average_pooling2d_1/PartitionedCall�
)res_conv_A_repr_d/StatefulPartitionedCallStatefulPartitionedCall,average_pooling2d_1/PartitionedCall:output:00res_conv_a_repr_d_statefulpartitionedcall_args_10res_conv_a_repr_d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_A_repr_d_layer_call_and_return_conditional_losses_228872+
)res_conv_A_repr_d/StatefulPartitionedCall�
activation_6/PartitionedCallPartitionedCall2res_conv_A_repr_d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_6_layer_call_and_return_conditional_losses_239052
activation_6/PartitionedCall�
-batch_normalization_6/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:04batch_normalization_6_statefulpartitionedcall_args_14batch_normalization_6_statefulpartitionedcall_args_24batch_normalization_6_statefulpartitionedcall_args_34batch_normalization_6_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_239462/
-batch_normalization_6/StatefulPartitionedCall�
)res_conv_B_repr_d/StatefulPartitionedCallStatefulPartitionedCall6batch_normalization_6/StatefulPartitionedCall:output:00res_conv_b_repr_d_statefulpartitionedcall_args_10res_conv_b_repr_d_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_B_repr_d_layer_call_and_return_conditional_losses_230472+
)res_conv_B_repr_d/StatefulPartitionedCall�
activation_7/PartitionedCallPartitionedCall2res_conv_B_repr_d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_7_layer_call_and_return_conditional_losses_240002
activation_7/PartitionedCall�
-batch_normalization_7/StatefulPartitionedCallStatefulPartitionedCall%activation_7/PartitionedCall:output:04batch_normalization_7_statefulpartitionedcall_args_14batch_normalization_7_statefulpartitionedcall_args_24batch_normalization_7_statefulpartitionedcall_args_34batch_normalization_7_statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_240412/
-batch_normalization_7/StatefulPartitionedCall�
add_3/PartitionedCallPartitionedCall6batch_normalization_7/StatefulPartitionedCall:output:0,average_pooling2d_1/PartitionedCall:output:0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*I
fDRB
@__inference_add_3_layer_call_and_return_conditional_losses_240932
add_3/PartitionedCall�
"repr_board/StatefulPartitionedCallStatefulPartitionedCalladd_3/PartitionedCall:output:0)repr_board_statefulpartitionedcall_args_1)repr_board_statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������*/
config_proto

GPU

CPU2 *0J 8*N
fIRG
E__inference_repr_board_layer_call_and_return_conditional_losses_232162$
"repr_board/StatefulPartitionedCall�
:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_a_repr_a_statefulpartitionedcall_args_1*^res_conv_A_repr_a/StatefulPartitionedCall*&
_output_shapes
:  *
dtype02<
:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_a/kernel/Regularizer/SquareSquareBres_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2-
+res_conv_A_repr_a/kernel/Regularizer/Square�
*res_conv_A_repr_a/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_a/kernel/Regularizer/Const�
(res_conv_A_repr_a/kernel/Regularizer/SumSum/res_conv_A_repr_a/kernel/Regularizer/Square:y:03res_conv_A_repr_a/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_a/kernel/Regularizer/Sum�
*res_conv_A_repr_a/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_a/kernel/Regularizer/mul/x�
(res_conv_A_repr_a/kernel/Regularizer/mulMul3res_conv_A_repr_a/kernel/Regularizer/mul/x:output:01res_conv_A_repr_a/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_a/kernel/Regularizer/mul�
*res_conv_A_repr_a/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_a/kernel/Regularizer/add/x�
(res_conv_A_repr_a/kernel/Regularizer/addAddV23res_conv_A_repr_a/kernel/Regularizer/add/x:output:0,res_conv_A_repr_a/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_a/kernel/Regularizer/add�
:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_b_repr_a_statefulpartitionedcall_args_1*^res_conv_B_repr_a/StatefulPartitionedCall*&
_output_shapes
:  *
dtype02<
:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_a/kernel/Regularizer/SquareSquareBres_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2-
+res_conv_B_repr_a/kernel/Regularizer/Square�
*res_conv_B_repr_a/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_a/kernel/Regularizer/Const�
(res_conv_B_repr_a/kernel/Regularizer/SumSum/res_conv_B_repr_a/kernel/Regularizer/Square:y:03res_conv_B_repr_a/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_a/kernel/Regularizer/Sum�
*res_conv_B_repr_a/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_a/kernel/Regularizer/mul/x�
(res_conv_B_repr_a/kernel/Regularizer/mulMul3res_conv_B_repr_a/kernel/Regularizer/mul/x:output:01res_conv_B_repr_a/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_a/kernel/Regularizer/mul�
*res_conv_B_repr_a/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_a/kernel/Regularizer/add/x�
(res_conv_B_repr_a/kernel/Regularizer/addAddV23res_conv_B_repr_a/kernel/Regularizer/add/x:output:0,res_conv_B_repr_a/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_a/kernel/Regularizer/add�
:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_a_repr_b_statefulpartitionedcall_args_1*^res_conv_A_repr_b/StatefulPartitionedCall*&
_output_shapes
:@@*
dtype02<
:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_b/kernel/Regularizer/SquareSquareBres_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_A_repr_b/kernel/Regularizer/Square�
*res_conv_A_repr_b/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_b/kernel/Regularizer/Const�
(res_conv_A_repr_b/kernel/Regularizer/SumSum/res_conv_A_repr_b/kernel/Regularizer/Square:y:03res_conv_A_repr_b/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_b/kernel/Regularizer/Sum�
*res_conv_A_repr_b/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_b/kernel/Regularizer/mul/x�
(res_conv_A_repr_b/kernel/Regularizer/mulMul3res_conv_A_repr_b/kernel/Regularizer/mul/x:output:01res_conv_A_repr_b/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_b/kernel/Regularizer/mul�
*res_conv_A_repr_b/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_b/kernel/Regularizer/add/x�
(res_conv_A_repr_b/kernel/Regularizer/addAddV23res_conv_A_repr_b/kernel/Regularizer/add/x:output:0,res_conv_A_repr_b/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_b/kernel/Regularizer/add�
:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_b_repr_b_statefulpartitionedcall_args_1*^res_conv_B_repr_b/StatefulPartitionedCall*&
_output_shapes
:@@*
dtype02<
:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_b/kernel/Regularizer/SquareSquareBres_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_B_repr_b/kernel/Regularizer/Square�
*res_conv_B_repr_b/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_b/kernel/Regularizer/Const�
(res_conv_B_repr_b/kernel/Regularizer/SumSum/res_conv_B_repr_b/kernel/Regularizer/Square:y:03res_conv_B_repr_b/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_b/kernel/Regularizer/Sum�
*res_conv_B_repr_b/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_b/kernel/Regularizer/mul/x�
(res_conv_B_repr_b/kernel/Regularizer/mulMul3res_conv_B_repr_b/kernel/Regularizer/mul/x:output:01res_conv_B_repr_b/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_b/kernel/Regularizer/mul�
*res_conv_B_repr_b/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_b/kernel/Regularizer/add/x�
(res_conv_B_repr_b/kernel/Regularizer/addAddV23res_conv_B_repr_b/kernel/Regularizer/add/x:output:0,res_conv_B_repr_b/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_b/kernel/Regularizer/add�
:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_a_repr_c_statefulpartitionedcall_args_1*^res_conv_A_repr_c/StatefulPartitionedCall*&
_output_shapes
:@@*
dtype02<
:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_c/kernel/Regularizer/SquareSquareBres_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_A_repr_c/kernel/Regularizer/Square�
*res_conv_A_repr_c/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_c/kernel/Regularizer/Const�
(res_conv_A_repr_c/kernel/Regularizer/SumSum/res_conv_A_repr_c/kernel/Regularizer/Square:y:03res_conv_A_repr_c/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_c/kernel/Regularizer/Sum�
*res_conv_A_repr_c/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_c/kernel/Regularizer/mul/x�
(res_conv_A_repr_c/kernel/Regularizer/mulMul3res_conv_A_repr_c/kernel/Regularizer/mul/x:output:01res_conv_A_repr_c/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_c/kernel/Regularizer/mul�
*res_conv_A_repr_c/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_c/kernel/Regularizer/add/x�
(res_conv_A_repr_c/kernel/Regularizer/addAddV23res_conv_A_repr_c/kernel/Regularizer/add/x:output:0,res_conv_A_repr_c/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_c/kernel/Regularizer/add�
:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_b_repr_c_statefulpartitionedcall_args_1*^res_conv_B_repr_c/StatefulPartitionedCall*&
_output_shapes
:@@*
dtype02<
:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_c/kernel/Regularizer/SquareSquareBres_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_B_repr_c/kernel/Regularizer/Square�
*res_conv_B_repr_c/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_c/kernel/Regularizer/Const�
(res_conv_B_repr_c/kernel/Regularizer/SumSum/res_conv_B_repr_c/kernel/Regularizer/Square:y:03res_conv_B_repr_c/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_c/kernel/Regularizer/Sum�
*res_conv_B_repr_c/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_c/kernel/Regularizer/mul/x�
(res_conv_B_repr_c/kernel/Regularizer/mulMul3res_conv_B_repr_c/kernel/Regularizer/mul/x:output:01res_conv_B_repr_c/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_c/kernel/Regularizer/mul�
*res_conv_B_repr_c/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_c/kernel/Regularizer/add/x�
(res_conv_B_repr_c/kernel/Regularizer/addAddV23res_conv_B_repr_c/kernel/Regularizer/add/x:output:0,res_conv_B_repr_c/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_c/kernel/Regularizer/add�
:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_a_repr_d_statefulpartitionedcall_args_1*^res_conv_A_repr_d/StatefulPartitionedCall*&
_output_shapes
:@@*
dtype02<
:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_d/kernel/Regularizer/SquareSquareBres_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_A_repr_d/kernel/Regularizer/Square�
*res_conv_A_repr_d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_d/kernel/Regularizer/Const�
(res_conv_A_repr_d/kernel/Regularizer/SumSum/res_conv_A_repr_d/kernel/Regularizer/Square:y:03res_conv_A_repr_d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_d/kernel/Regularizer/Sum�
*res_conv_A_repr_d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_d/kernel/Regularizer/mul/x�
(res_conv_A_repr_d/kernel/Regularizer/mulMul3res_conv_A_repr_d/kernel/Regularizer/mul/x:output:01res_conv_A_repr_d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_d/kernel/Regularizer/mul�
*res_conv_A_repr_d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_d/kernel/Regularizer/add/x�
(res_conv_A_repr_d/kernel/Regularizer/addAddV23res_conv_A_repr_d/kernel/Regularizer/add/x:output:0,res_conv_A_repr_d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_d/kernel/Regularizer/add�
:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOpReadVariableOp0res_conv_b_repr_d_statefulpartitionedcall_args_1*^res_conv_B_repr_d/StatefulPartitionedCall*&
_output_shapes
:@@*
dtype02<
:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_d/kernel/Regularizer/SquareSquareBres_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_B_repr_d/kernel/Regularizer/Square�
*res_conv_B_repr_d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_d/kernel/Regularizer/Const�
(res_conv_B_repr_d/kernel/Regularizer/SumSum/res_conv_B_repr_d/kernel/Regularizer/Square:y:03res_conv_B_repr_d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_d/kernel/Regularizer/Sum�
*res_conv_B_repr_d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_d/kernel/Regularizer/mul/x�
(res_conv_B_repr_d/kernel/Regularizer/mulMul3res_conv_B_repr_d/kernel/Regularizer/mul/x:output:01res_conv_B_repr_d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_d/kernel/Regularizer/mul�
*res_conv_B_repr_d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_d/kernel/Regularizer/add/x�
(res_conv_B_repr_d/kernel/Regularizer/addAddV23res_conv_B_repr_d/kernel/Regularizer/add/x:output:0,res_conv_B_repr_d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_d/kernel/Regularizer/add�
3repr_board/kernel/Regularizer/Square/ReadVariableOpReadVariableOp)repr_board_statefulpartitionedcall_args_1#^repr_board/StatefulPartitionedCall*&
_output_shapes
:@*
dtype025
3repr_board/kernel/Regularizer/Square/ReadVariableOp�
$repr_board/kernel/Regularizer/SquareSquare;repr_board/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@2&
$repr_board/kernel/Regularizer/Square�
#repr_board/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2%
#repr_board/kernel/Regularizer/Const�
!repr_board/kernel/Regularizer/SumSum(repr_board/kernel/Regularizer/Square:y:0,repr_board/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2#
!repr_board/kernel/Regularizer/Sum�
#repr_board/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72%
#repr_board/kernel/Regularizer/mul/x�
!repr_board/kernel/Regularizer/mulMul,repr_board/kernel/Regularizer/mul/x:output:0*repr_board/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2#
!repr_board/kernel/Regularizer/mul�
#repr_board/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#repr_board/kernel/Regularizer/add/x�
!repr_board/kernel/Regularizer/addAddV2,repr_board/kernel/Regularizer/add/x:output:0%repr_board/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2#
!repr_board/kernel/Regularizer/add�
1repr_board/bias/Regularizer/Square/ReadVariableOpReadVariableOp)repr_board_statefulpartitionedcall_args_2#^repr_board/StatefulPartitionedCall*
_output_shapes
:*
dtype023
1repr_board/bias/Regularizer/Square/ReadVariableOp�
"repr_board/bias/Regularizer/SquareSquare9repr_board/bias/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes
:2$
"repr_board/bias/Regularizer/Square�
!repr_board/bias/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2#
!repr_board/bias/Regularizer/Const�
repr_board/bias/Regularizer/SumSum&repr_board/bias/Regularizer/Square:y:0*repr_board/bias/Regularizer/Const:output:0*
T0*
_output_shapes
: 2!
repr_board/bias/Regularizer/Sum�
!repr_board/bias/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72#
!repr_board/bias/Regularizer/mul/x�
repr_board/bias/Regularizer/mulMul*repr_board/bias/Regularizer/mul/x:output:0(repr_board/bias/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2!
repr_board/bias/Regularizer/mul�
!repr_board/bias/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!repr_board/bias/Regularizer/add/x�
repr_board/bias/Regularizer/addAddV2*repr_board/bias/Regularizer/add/x:output:0#repr_board/bias/Regularizer/mul:z:0*
T0*
_output_shapes
: 2!
repr_board/bias/Regularizer/add�
IdentityIdentity+repr_board/StatefulPartitionedCall:output:0,^batch_normalization/StatefulPartitionedCall.^batch_normalization_1/StatefulPartitionedCall.^batch_normalization_2/StatefulPartitionedCall.^batch_normalization_3/StatefulPartitionedCall.^batch_normalization_4/StatefulPartitionedCall.^batch_normalization_5/StatefulPartitionedCall.^batch_normalization_6/StatefulPartitionedCall.^batch_normalization_7/StatefulPartitionedCall^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall#^repr_board/StatefulPartitionedCall2^repr_board/bias/Regularizer/Square/ReadVariableOp4^repr_board/kernel/Regularizer/Square/ReadVariableOp*^res_conv_A_repr_a/StatefulPartitionedCall;^res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp*^res_conv_A_repr_b/StatefulPartitionedCall;^res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp*^res_conv_A_repr_c/StatefulPartitionedCall;^res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp*^res_conv_A_repr_d/StatefulPartitionedCall;^res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp*^res_conv_B_repr_a/StatefulPartitionedCall;^res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp*^res_conv_B_repr_b/StatefulPartitionedCall;^res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp*^res_conv_B_repr_c/StatefulPartitionedCall;^res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp*^res_conv_B_repr_d/StatefulPartitionedCall;^res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp*
T0*/
_output_shapes
:���������2

Identity"
identityIdentity:output:0*�
_input_shapes�
�:���������``::::::::::::::::::::::::::::::::::::::::::::::::::::::2Z
+batch_normalization/StatefulPartitionedCall+batch_normalization/StatefulPartitionedCall2^
-batch_normalization_1/StatefulPartitionedCall-batch_normalization_1/StatefulPartitionedCall2^
-batch_normalization_2/StatefulPartitionedCall-batch_normalization_2/StatefulPartitionedCall2^
-batch_normalization_3/StatefulPartitionedCall-batch_normalization_3/StatefulPartitionedCall2^
-batch_normalization_4/StatefulPartitionedCall-batch_normalization_4/StatefulPartitionedCall2^
-batch_normalization_5/StatefulPartitionedCall-batch_normalization_5/StatefulPartitionedCall2^
-batch_normalization_6/StatefulPartitionedCall-batch_normalization_6/StatefulPartitionedCall2^
-batch_normalization_7/StatefulPartitionedCall-batch_normalization_7/StatefulPartitionedCall2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2H
"repr_board/StatefulPartitionedCall"repr_board/StatefulPartitionedCall2f
1repr_board/bias/Regularizer/Square/ReadVariableOp1repr_board/bias/Regularizer/Square/ReadVariableOp2j
3repr_board/kernel/Regularizer/Square/ReadVariableOp3repr_board/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_A_repr_a/StatefulPartitionedCall)res_conv_A_repr_a/StatefulPartitionedCall2x
:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_A_repr_b/StatefulPartitionedCall)res_conv_A_repr_b/StatefulPartitionedCall2x
:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_A_repr_c/StatefulPartitionedCall)res_conv_A_repr_c/StatefulPartitionedCall2x
:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_c/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_A_repr_d/StatefulPartitionedCall)res_conv_A_repr_d/StatefulPartitionedCall2x
:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_d/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_B_repr_a/StatefulPartitionedCall)res_conv_B_repr_a/StatefulPartitionedCall2x
:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_B_repr_b/StatefulPartitionedCall)res_conv_B_repr_b/StatefulPartitionedCall2x
:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_b/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_B_repr_c/StatefulPartitionedCall)res_conv_B_repr_c/StatefulPartitionedCall2x
:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_c/kernel/Regularizer/Square/ReadVariableOp2V
)res_conv_B_repr_d/StatefulPartitionedCall)res_conv_B_repr_d/StatefulPartitionedCall2x
:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp:% !

_user_specified_nameboard
�$
�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22325

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_22310
assignmovingavg_1_22317
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/22310*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/22310*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_22310*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/22310*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/22310*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_22310AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/22310*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/22317*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/22317*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_22317*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/22317*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/22317*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_22317AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/22317*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�

�
A__inference_conv2d_layer_call_and_return_conditional_losses_21855

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� 2	
BiasAdd�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
L__inference_res_conv_B_repr_d_layer_call_and_return_conditional_losses_23047

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAdd�
:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource^Conv2D/ReadVariableOp*&
_output_shapes
:@@*
dtype02<
:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_d/kernel/Regularizer/SquareSquareBres_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_B_repr_d/kernel/Regularizer/Square�
*res_conv_B_repr_d/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_d/kernel/Regularizer/Const�
(res_conv_B_repr_d/kernel/Regularizer/SumSum/res_conv_B_repr_d/kernel/Regularizer/Square:y:03res_conv_B_repr_d/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_d/kernel/Regularizer/Sum�
*res_conv_B_repr_d/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_d/kernel/Regularizer/mul/x�
(res_conv_B_repr_d/kernel/Regularizer/mulMul3res_conv_B_repr_d/kernel/Regularizer/mul/x:output:01res_conv_B_repr_d/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_d/kernel/Regularizer/mul�
*res_conv_B_repr_d/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_d/kernel/Regularizer/add/x�
(res_conv_B_repr_d/kernel/Regularizer/addAddV23res_conv_B_repr_d/kernel/Regularizer/add/x:output:0,res_conv_B_repr_d/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_d/kernel/Regularizer/add�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp;^res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2x
:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_d/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_27494G
Cres_conv_b_repr_a_kernel_regularizer_square_readvariableop_resource
identity��:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp�
:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOpReadVariableOpCres_conv_b_repr_a_kernel_regularizer_square_readvariableop_resource*&
_output_shapes
:  *
dtype02<
:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_B_repr_a/kernel/Regularizer/SquareSquareBres_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2-
+res_conv_B_repr_a/kernel/Regularizer/Square�
*res_conv_B_repr_a/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_B_repr_a/kernel/Regularizer/Const�
(res_conv_B_repr_a/kernel/Regularizer/SumSum/res_conv_B_repr_a/kernel/Regularizer/Square:y:03res_conv_B_repr_a/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_a/kernel/Regularizer/Sum�
*res_conv_B_repr_a/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_B_repr_a/kernel/Regularizer/mul/x�
(res_conv_B_repr_a/kernel/Regularizer/mulMul3res_conv_B_repr_a/kernel/Regularizer/mul/x:output:01res_conv_B_repr_a/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_a/kernel/Regularizer/mul�
*res_conv_B_repr_a/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_B_repr_a/kernel/Regularizer/add/x�
(res_conv_B_repr_a/kernel/Regularizer/addAddV23res_conv_B_repr_a/kernel/Regularizer/add/x:output:0,res_conv_B_repr_a/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_B_repr_a/kernel/Regularizer/add�
IdentityIdentity,res_conv_B_repr_a/kernel/Regularizer/add:z:0;^res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2x
:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp:res_conv_B_repr_a/kernel/Regularizer/Square/ReadVariableOp
�
�
(__inference_conv2d_1_layer_call_fn_22203

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

GPU

CPU2 *0J 8*L
fGRE
C__inference_conv2d_1_layer_call_and_return_conditional_losses_221952
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_5_layer_call_fn_27072

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_228482
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26802

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�#
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_26222

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_26207
assignmovingavg_1_26214
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������00 : : : : :*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/26207*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/26207*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_26207*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/26207*
_output_shapes
: 2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/26207*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_26207AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/26207*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/26214*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/26214*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_26214*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/26214*
_output_shapes
: 2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/26214*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_26214AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/26214*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������00 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������00 ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
��
�
!__inference__traced_restore_27958
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias/
+assignvariableop_2_res_conv_a_repr_a_kernel-
)assignvariableop_3_res_conv_a_repr_a_bias0
,assignvariableop_4_batch_normalization_gamma/
+assignvariableop_5_batch_normalization_beta6
2assignvariableop_6_batch_normalization_moving_mean:
6assignvariableop_7_batch_normalization_moving_variance/
+assignvariableop_8_res_conv_b_repr_a_kernel-
)assignvariableop_9_res_conv_b_repr_a_bias3
/assignvariableop_10_batch_normalization_1_gamma2
.assignvariableop_11_batch_normalization_1_beta9
5assignvariableop_12_batch_normalization_1_moving_mean=
9assignvariableop_13_batch_normalization_1_moving_variance'
#assignvariableop_14_conv2d_1_kernel%
!assignvariableop_15_conv2d_1_bias0
,assignvariableop_16_res_conv_a_repr_b_kernel.
*assignvariableop_17_res_conv_a_repr_b_bias3
/assignvariableop_18_batch_normalization_2_gamma2
.assignvariableop_19_batch_normalization_2_beta9
5assignvariableop_20_batch_normalization_2_moving_mean=
9assignvariableop_21_batch_normalization_2_moving_variance0
,assignvariableop_22_res_conv_b_repr_b_kernel.
*assignvariableop_23_res_conv_b_repr_b_bias3
/assignvariableop_24_batch_normalization_3_gamma2
.assignvariableop_25_batch_normalization_3_beta9
5assignvariableop_26_batch_normalization_3_moving_mean=
9assignvariableop_27_batch_normalization_3_moving_variance0
,assignvariableop_28_res_conv_a_repr_c_kernel.
*assignvariableop_29_res_conv_a_repr_c_bias3
/assignvariableop_30_batch_normalization_4_gamma2
.assignvariableop_31_batch_normalization_4_beta9
5assignvariableop_32_batch_normalization_4_moving_mean=
9assignvariableop_33_batch_normalization_4_moving_variance0
,assignvariableop_34_res_conv_b_repr_c_kernel.
*assignvariableop_35_res_conv_b_repr_c_bias3
/assignvariableop_36_batch_normalization_5_gamma2
.assignvariableop_37_batch_normalization_5_beta9
5assignvariableop_38_batch_normalization_5_moving_mean=
9assignvariableop_39_batch_normalization_5_moving_variance0
,assignvariableop_40_res_conv_a_repr_d_kernel.
*assignvariableop_41_res_conv_a_repr_d_bias3
/assignvariableop_42_batch_normalization_6_gamma2
.assignvariableop_43_batch_normalization_6_beta9
5assignvariableop_44_batch_normalization_6_moving_mean=
9assignvariableop_45_batch_normalization_6_moving_variance0
,assignvariableop_46_res_conv_b_repr_d_kernel.
*assignvariableop_47_res_conv_b_repr_d_bias3
/assignvariableop_48_batch_normalization_7_gamma2
.assignvariableop_49_batch_normalization_7_beta9
5assignvariableop_50_batch_normalization_7_moving_mean=
9assignvariableop_51_batch_normalization_7_moving_variance)
%assignvariableop_52_repr_board_kernel'
#assignvariableop_53_repr_board_bias
identity_55��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_27�AssignVariableOp_28�AssignVariableOp_29�AssignVariableOp_3�AssignVariableOp_30�AssignVariableOp_31�AssignVariableOp_32�AssignVariableOp_33�AssignVariableOp_34�AssignVariableOp_35�AssignVariableOp_36�AssignVariableOp_37�AssignVariableOp_38�AssignVariableOp_39�AssignVariableOp_4�AssignVariableOp_40�AssignVariableOp_41�AssignVariableOp_42�AssignVariableOp_43�AssignVariableOp_44�AssignVariableOp_45�AssignVariableOp_46�AssignVariableOp_47�AssignVariableOp_48�AssignVariableOp_49�AssignVariableOp_5�AssignVariableOp_50�AssignVariableOp_51�AssignVariableOp_52�AssignVariableOp_53�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�	RestoreV2�RestoreV2_1�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*�
value�B�6B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-4/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-4/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-4/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-9/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-9/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-11/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-11/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-13/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-13/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-13/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-13/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-14/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-14/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-15/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-15/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-15/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-15/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-16/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-16/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-17/gamma/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-17/beta/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-17/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB@layer_with_weights-17/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-18/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-18/bias/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:6*
dtype0*
valuevBt6B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::*D
dtypes:
8262
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOp+assignvariableop_2_res_conv_a_repr_a_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOp)assignvariableop_3_res_conv_a_repr_a_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3\

Identity_4IdentityRestoreV2:tensors:4*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOp,assignvariableop_4_batch_normalization_gammaIdentity_4:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_4\

Identity_5IdentityRestoreV2:tensors:5*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOp+assignvariableop_5_batch_normalization_betaIdentity_5:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_5\

Identity_6IdentityRestoreV2:tensors:6*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp2assignvariableop_6_batch_normalization_moving_meanIdentity_6:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_6\

Identity_7IdentityRestoreV2:tensors:7*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOp6assignvariableop_7_batch_normalization_moving_varianceIdentity_7:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_7\

Identity_8IdentityRestoreV2:tensors:8*
T0*
_output_shapes
:2

Identity_8�
AssignVariableOp_8AssignVariableOp+assignvariableop_8_res_conv_b_repr_a_kernelIdentity_8:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_8\

Identity_9IdentityRestoreV2:tensors:9*
T0*
_output_shapes
:2

Identity_9�
AssignVariableOp_9AssignVariableOp)assignvariableop_9_res_conv_b_repr_a_biasIdentity_9:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_9_
Identity_10IdentityRestoreV2:tensors:10*
T0*
_output_shapes
:2
Identity_10�
AssignVariableOp_10AssignVariableOp/assignvariableop_10_batch_normalization_1_gammaIdentity_10:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_10_
Identity_11IdentityRestoreV2:tensors:11*
T0*
_output_shapes
:2
Identity_11�
AssignVariableOp_11AssignVariableOp.assignvariableop_11_batch_normalization_1_betaIdentity_11:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_11_
Identity_12IdentityRestoreV2:tensors:12*
T0*
_output_shapes
:2
Identity_12�
AssignVariableOp_12AssignVariableOp5assignvariableop_12_batch_normalization_1_moving_meanIdentity_12:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_12_
Identity_13IdentityRestoreV2:tensors:13*
T0*
_output_shapes
:2
Identity_13�
AssignVariableOp_13AssignVariableOp9assignvariableop_13_batch_normalization_1_moving_varianceIdentity_13:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_13_
Identity_14IdentityRestoreV2:tensors:14*
T0*
_output_shapes
:2
Identity_14�
AssignVariableOp_14AssignVariableOp#assignvariableop_14_conv2d_1_kernelIdentity_14:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_14_
Identity_15IdentityRestoreV2:tensors:15*
T0*
_output_shapes
:2
Identity_15�
AssignVariableOp_15AssignVariableOp!assignvariableop_15_conv2d_1_biasIdentity_15:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_15_
Identity_16IdentityRestoreV2:tensors:16*
T0*
_output_shapes
:2
Identity_16�
AssignVariableOp_16AssignVariableOp,assignvariableop_16_res_conv_a_repr_b_kernelIdentity_16:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_16_
Identity_17IdentityRestoreV2:tensors:17*
T0*
_output_shapes
:2
Identity_17�
AssignVariableOp_17AssignVariableOp*assignvariableop_17_res_conv_a_repr_b_biasIdentity_17:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_17_
Identity_18IdentityRestoreV2:tensors:18*
T0*
_output_shapes
:2
Identity_18�
AssignVariableOp_18AssignVariableOp/assignvariableop_18_batch_normalization_2_gammaIdentity_18:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_18_
Identity_19IdentityRestoreV2:tensors:19*
T0*
_output_shapes
:2
Identity_19�
AssignVariableOp_19AssignVariableOp.assignvariableop_19_batch_normalization_2_betaIdentity_19:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_19_
Identity_20IdentityRestoreV2:tensors:20*
T0*
_output_shapes
:2
Identity_20�
AssignVariableOp_20AssignVariableOp5assignvariableop_20_batch_normalization_2_moving_meanIdentity_20:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_20_
Identity_21IdentityRestoreV2:tensors:21*
T0*
_output_shapes
:2
Identity_21�
AssignVariableOp_21AssignVariableOp9assignvariableop_21_batch_normalization_2_moving_varianceIdentity_21:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_21_
Identity_22IdentityRestoreV2:tensors:22*
T0*
_output_shapes
:2
Identity_22�
AssignVariableOp_22AssignVariableOp,assignvariableop_22_res_conv_b_repr_b_kernelIdentity_22:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_22_
Identity_23IdentityRestoreV2:tensors:23*
T0*
_output_shapes
:2
Identity_23�
AssignVariableOp_23AssignVariableOp*assignvariableop_23_res_conv_b_repr_b_biasIdentity_23:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_23_
Identity_24IdentityRestoreV2:tensors:24*
T0*
_output_shapes
:2
Identity_24�
AssignVariableOp_24AssignVariableOp/assignvariableop_24_batch_normalization_3_gammaIdentity_24:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_24_
Identity_25IdentityRestoreV2:tensors:25*
T0*
_output_shapes
:2
Identity_25�
AssignVariableOp_25AssignVariableOp.assignvariableop_25_batch_normalization_3_betaIdentity_25:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_25_
Identity_26IdentityRestoreV2:tensors:26*
T0*
_output_shapes
:2
Identity_26�
AssignVariableOp_26AssignVariableOp5assignvariableop_26_batch_normalization_3_moving_meanIdentity_26:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_26_
Identity_27IdentityRestoreV2:tensors:27*
T0*
_output_shapes
:2
Identity_27�
AssignVariableOp_27AssignVariableOp9assignvariableop_27_batch_normalization_3_moving_varianceIdentity_27:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_27_
Identity_28IdentityRestoreV2:tensors:28*
T0*
_output_shapes
:2
Identity_28�
AssignVariableOp_28AssignVariableOp,assignvariableop_28_res_conv_a_repr_c_kernelIdentity_28:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_28_
Identity_29IdentityRestoreV2:tensors:29*
T0*
_output_shapes
:2
Identity_29�
AssignVariableOp_29AssignVariableOp*assignvariableop_29_res_conv_a_repr_c_biasIdentity_29:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_29_
Identity_30IdentityRestoreV2:tensors:30*
T0*
_output_shapes
:2
Identity_30�
AssignVariableOp_30AssignVariableOp/assignvariableop_30_batch_normalization_4_gammaIdentity_30:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_30_
Identity_31IdentityRestoreV2:tensors:31*
T0*
_output_shapes
:2
Identity_31�
AssignVariableOp_31AssignVariableOp.assignvariableop_31_batch_normalization_4_betaIdentity_31:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_31_
Identity_32IdentityRestoreV2:tensors:32*
T0*
_output_shapes
:2
Identity_32�
AssignVariableOp_32AssignVariableOp5assignvariableop_32_batch_normalization_4_moving_meanIdentity_32:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_32_
Identity_33IdentityRestoreV2:tensors:33*
T0*
_output_shapes
:2
Identity_33�
AssignVariableOp_33AssignVariableOp9assignvariableop_33_batch_normalization_4_moving_varianceIdentity_33:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_33_
Identity_34IdentityRestoreV2:tensors:34*
T0*
_output_shapes
:2
Identity_34�
AssignVariableOp_34AssignVariableOp,assignvariableop_34_res_conv_b_repr_c_kernelIdentity_34:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_34_
Identity_35IdentityRestoreV2:tensors:35*
T0*
_output_shapes
:2
Identity_35�
AssignVariableOp_35AssignVariableOp*assignvariableop_35_res_conv_b_repr_c_biasIdentity_35:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_35_
Identity_36IdentityRestoreV2:tensors:36*
T0*
_output_shapes
:2
Identity_36�
AssignVariableOp_36AssignVariableOp/assignvariableop_36_batch_normalization_5_gammaIdentity_36:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_36_
Identity_37IdentityRestoreV2:tensors:37*
T0*
_output_shapes
:2
Identity_37�
AssignVariableOp_37AssignVariableOp.assignvariableop_37_batch_normalization_5_betaIdentity_37:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_37_
Identity_38IdentityRestoreV2:tensors:38*
T0*
_output_shapes
:2
Identity_38�
AssignVariableOp_38AssignVariableOp5assignvariableop_38_batch_normalization_5_moving_meanIdentity_38:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_38_
Identity_39IdentityRestoreV2:tensors:39*
T0*
_output_shapes
:2
Identity_39�
AssignVariableOp_39AssignVariableOp9assignvariableop_39_batch_normalization_5_moving_varianceIdentity_39:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_39_
Identity_40IdentityRestoreV2:tensors:40*
T0*
_output_shapes
:2
Identity_40�
AssignVariableOp_40AssignVariableOp,assignvariableop_40_res_conv_a_repr_d_kernelIdentity_40:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_40_
Identity_41IdentityRestoreV2:tensors:41*
T0*
_output_shapes
:2
Identity_41�
AssignVariableOp_41AssignVariableOp*assignvariableop_41_res_conv_a_repr_d_biasIdentity_41:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_41_
Identity_42IdentityRestoreV2:tensors:42*
T0*
_output_shapes
:2
Identity_42�
AssignVariableOp_42AssignVariableOp/assignvariableop_42_batch_normalization_6_gammaIdentity_42:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_42_
Identity_43IdentityRestoreV2:tensors:43*
T0*
_output_shapes
:2
Identity_43�
AssignVariableOp_43AssignVariableOp.assignvariableop_43_batch_normalization_6_betaIdentity_43:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_43_
Identity_44IdentityRestoreV2:tensors:44*
T0*
_output_shapes
:2
Identity_44�
AssignVariableOp_44AssignVariableOp5assignvariableop_44_batch_normalization_6_moving_meanIdentity_44:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_44_
Identity_45IdentityRestoreV2:tensors:45*
T0*
_output_shapes
:2
Identity_45�
AssignVariableOp_45AssignVariableOp9assignvariableop_45_batch_normalization_6_moving_varianceIdentity_45:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_45_
Identity_46IdentityRestoreV2:tensors:46*
T0*
_output_shapes
:2
Identity_46�
AssignVariableOp_46AssignVariableOp,assignvariableop_46_res_conv_b_repr_d_kernelIdentity_46:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_46_
Identity_47IdentityRestoreV2:tensors:47*
T0*
_output_shapes
:2
Identity_47�
AssignVariableOp_47AssignVariableOp*assignvariableop_47_res_conv_b_repr_d_biasIdentity_47:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_47_
Identity_48IdentityRestoreV2:tensors:48*
T0*
_output_shapes
:2
Identity_48�
AssignVariableOp_48AssignVariableOp/assignvariableop_48_batch_normalization_7_gammaIdentity_48:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_48_
Identity_49IdentityRestoreV2:tensors:49*
T0*
_output_shapes
:2
Identity_49�
AssignVariableOp_49AssignVariableOp.assignvariableop_49_batch_normalization_7_betaIdentity_49:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_49_
Identity_50IdentityRestoreV2:tensors:50*
T0*
_output_shapes
:2
Identity_50�
AssignVariableOp_50AssignVariableOp5assignvariableop_50_batch_normalization_7_moving_meanIdentity_50:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_50_
Identity_51IdentityRestoreV2:tensors:51*
T0*
_output_shapes
:2
Identity_51�
AssignVariableOp_51AssignVariableOp9assignvariableop_51_batch_normalization_7_moving_varianceIdentity_51:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_51_
Identity_52IdentityRestoreV2:tensors:52*
T0*
_output_shapes
:2
Identity_52�
AssignVariableOp_52AssignVariableOp%assignvariableop_52_repr_board_kernelIdentity_52:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_52_
Identity_53IdentityRestoreV2:tensors:53*
T0*
_output_shapes
:2
Identity_53�
AssignVariableOp_53AssignVariableOp#assignvariableop_53_repr_board_biasIdentity_53:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_53�
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_names�
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slices�
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_54Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_54�

Identity_55IdentityIdentity_54:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2
Identity_55"#
identity_55Identity_55:output:0*�
_input_shapes�
�: ::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
�$
�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_22485

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_22470
assignmovingavg_1_22477
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/22470*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/22470*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_22470*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/22470*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/22470*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_22470AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/22470*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/22477*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/22477*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_22477*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/22477*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/22477*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_22477AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/22477*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
L__inference_res_conv_A_repr_b_layer_call_and_return_conditional_losses_22223

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@*
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+���������������������������@2	
BiasAdd�
:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource^Conv2D/ReadVariableOp*&
_output_shapes
:@@*
dtype02<
:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_b/kernel/Regularizer/SquareSquareBres_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:@@2-
+res_conv_A_repr_b/kernel/Regularizer/Square�
*res_conv_A_repr_b/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_b/kernel/Regularizer/Const�
(res_conv_A_repr_b/kernel/Regularizer/SumSum/res_conv_A_repr_b/kernel/Regularizer/Square:y:03res_conv_A_repr_b/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_b/kernel/Regularizer/Sum�
*res_conv_A_repr_b/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_b/kernel/Regularizer/mul/x�
(res_conv_A_repr_b/kernel/Regularizer/mulMul3res_conv_A_repr_b/kernel/Regularizer/mul/x:output:01res_conv_A_repr_b/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_b/kernel/Regularizer/mul�
*res_conv_A_repr_b/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_b/kernel/Regularizer/add/x�
(res_conv_A_repr_b/kernel/Regularizer/addAddV23res_conv_A_repr_b/kernel/Regularizer/add/x:output:0,res_conv_A_repr_b/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_b/kernel/Regularizer/add�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp;^res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2x
:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_b/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_23651

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
j
N__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_22861

inputs
identity�
AvgPoolAvgPoolinputs*
T0*J
_output_shapes8
6:4������������������������������������*
ksize
*
paddingVALID*
strides
2	
AvgPool�
IdentityIdentityAvgPool:output:0*
T0*J
_output_shapes8
6:4������������������������������������2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4������������������������������������:& "
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_23556

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
H
,__inference_activation_7_layer_call_fn_27280

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*P
fKRI
G__inference_activation_7_layer_call_and_return_conditional_losses_240002
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23180

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�$
�
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26780

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_26765
assignmovingavg_1_26772
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/26765*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/26765*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_26765*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/26765*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/26765*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_26765AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/26765*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/26772*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/26772*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_26772*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/26772*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/26772*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_26772AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/26772*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_22356

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_27054

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_7_layer_call_fn_27357

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_231492
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_6_layer_call_fn_27262

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_239682
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_4_layer_call_fn_26885

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_237402
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
3__inference_batch_normalization_layer_call_fn_26158

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+��������������������������� */
config_proto

GPU

CPU2 *0J 8*W
fRRP
N__inference_batch_normalization_layer_call_and_return_conditional_losses_220162
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_26066

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������00 : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������00 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������00 ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
j
@__inference_add_3_layer_call_and_return_conditional_losses_24093

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������@:���������@:& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_7_layer_call_fn_27431

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_240412
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
c
G__inference_activation_3_layer_call_and_return_conditional_losses_23588

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�$
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_26296

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_26281
assignmovingavg_1_26288
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/26281*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/26281*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_26281*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/26281*
_output_shapes
: 2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/26281*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_26281AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/26281*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/26288*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/26288*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_26288*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/26288*
_output_shapes
: 2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/26288*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_26288AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/26288*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�#
�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26486

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_26471
assignmovingavg_1_26478
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/26471*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/26471*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_26471*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/26471*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/26471*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_26471AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/26471*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/26478*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/26478*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_26478*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/26478*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/26478*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_26478AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/26478*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_23020

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
l
@__inference_add_2_layer_call_and_return_conditional_losses_27078
inputs_0
inputs_1
identitya
addAddV2inputs_0inputs_1*
T0*/
_output_shapes
:���������@2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������@:���������@:( $
"
_user_specified_name
inputs/0:($
"
_user_specified_name
inputs/1
�#
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_26044

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_26029
assignmovingavg_1_26036
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������00 : : : : :*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/26029*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/26029*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_26029*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/26029*
_output_shapes
: 2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/26029*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_26029AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/26029*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/26036*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/26036*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_26036*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/26036*
_output_shapes
: 2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/26036*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_26036AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/26036*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������00 2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������00 ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_23762

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
L__inference_res_conv_A_repr_a_layer_call_and_return_conditional_losses_21883

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�Conv2D/ReadVariableOp�:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOpo
dilation_rateConst*
_output_shapes
:*
dtype0*
valueB"      2
dilation_rate�
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:  *
dtype02
Conv2D/ReadVariableOp�
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� *
paddingSAME*
strides
2
Conv2D�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+��������������������������� 2	
BiasAdd�
:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOpReadVariableOpconv2d_readvariableop_resource^Conv2D/ReadVariableOp*&
_output_shapes
:  *
dtype02<
:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp�
+res_conv_A_repr_a/kernel/Regularizer/SquareSquareBres_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp:value:0*
T0*&
_output_shapes
:  2-
+res_conv_A_repr_a/kernel/Regularizer/Square�
*res_conv_A_repr_a/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2,
*res_conv_A_repr_a/kernel/Regularizer/Const�
(res_conv_A_repr_a/kernel/Regularizer/SumSum/res_conv_A_repr_a/kernel/Regularizer/Square:y:03res_conv_A_repr_a/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_a/kernel/Regularizer/Sum�
*res_conv_A_repr_a/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *��'72,
*res_conv_A_repr_a/kernel/Regularizer/mul/x�
(res_conv_A_repr_a/kernel/Regularizer/mulMul3res_conv_A_repr_a/kernel/Regularizer/mul/x:output:01res_conv_A_repr_a/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_a/kernel/Regularizer/mul�
*res_conv_A_repr_a/kernel/Regularizer/add/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2,
*res_conv_A_repr_a/kernel/Regularizer/add/x�
(res_conv_A_repr_a/kernel/Regularizer/addAddV23res_conv_A_repr_a/kernel/Regularizer/add/x:output:0,res_conv_A_repr_a/kernel/Regularizer/mul:z:0*
T0*
_output_shapes
: 2*
(res_conv_A_repr_a/kernel/Regularizer/add�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp;^res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2x
:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp:res_conv_A_repr_a/kernel/Regularizer/Square/ReadVariableOp:& "
 
_user_specified_nameinputs
�
^
B__inference_reshape_layer_call_and_return_conditional_losses_25956

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :`2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :`2
Reshape/shape/3d
Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/4�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape{
ReshapeReshapeinputsReshape/shape:output:0*
T0*3
_output_shapes!
:���������``2	
Reshapep
IdentityIdentityReshape:output:0*
T0*3
_output_shapes!
:���������``2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������``:& "
 
_user_specified_nameinputs
�$
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_26118

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_26103
assignmovingavg_1_26110
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/26103*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/26103*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_26103*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/26103*
_output_shapes
: 2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/26103*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_26103AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/26103*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/26110*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/26110*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_26110*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/26110*
_output_shapes
: 2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/26110*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_26110AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/26110*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_22016

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
: *
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
: *
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
^
B__inference_reshape_layer_call_and_return_conditional_losses_23243

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :`2
Reshape/shape/2d
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B :`2
Reshape/shape/3d
Reshape/shape/4Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/4�
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0Reshape/shape/4:output:0*
N*
T0*
_output_shapes
:2
Reshape/shape{
ReshapeReshapeinputsReshape/shape:output:0*
T0*3
_output_shapes!
:���������``2	
Reshapep
IdentityIdentityReshape:output:0*
T0*3
_output_shapes!
:���������``2

Identity"
identityIdentity:output:0*2
_input_shapes!
:���������``:& "
 
_user_specified_nameinputs
�
�
1__inference_res_conv_B_repr_a_layer_call_fn_22051

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+��������������������������� */
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_B_repr_a_layer_call_and_return_conditional_losses_220432
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+��������������������������� ::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
c
G__inference_activation_1_layer_call_and_return_conditional_losses_26171

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������00 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������00 2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������00 :& "
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_5_layer_call_fn_26989

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_238352
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
c
G__inference_activation_4_layer_call_and_return_conditional_losses_26729

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_7_layer_call_fn_27440

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*/
_output_shapes
:���������@*/
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_240632
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�$
�
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_23149

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_23134
assignmovingavg_1_23141
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/23134*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/23134*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23134*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/23134*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/23134*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23134AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/23134*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/23141*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/23141*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23141*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/23141*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/23141*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23141AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/23141*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
�
1__inference_res_conv_A_repr_d_layer_call_fn_22895

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+���������������������������@*/
config_proto

GPU

CPU2 *0J 8*U
fPRN
L__inference_res_conv_A_repr_d_layer_call_and_return_conditional_losses_228872
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+���������������������������@::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�
^
B__inference_permute_layer_call_and_return_conditional_losses_21837

inputs
identity}
transpose/permConst*
_output_shapes
:*
dtype0*)
value B"                2
transpose/perm�
	transpose	Transposeinputstranspose/perm:output:0*
T0*W
_output_shapesE
C:A���������������������������������������������2
	transpose�
IdentityIdentitytranspose:y:0*
T0*W
_output_shapesE
C:A���������������������������������������������2

Identity"
identityIdentity:output:0*V
_input_shapesE
C:A���������������������������������������������:& "
 
_user_specified_nameinputs
�
�
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_22848

inputs
readvariableop_resource
readvariableop_1_resource,
(fusedbatchnormv3_readvariableop_resource.
*fusedbatchnormv3_readvariableop_1_resource
identity��FusedBatchNormV3/ReadVariableOp�!FusedBatchNormV3/ReadVariableOp_1�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z 2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1�
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype02!
FusedBatchNormV3/ReadVariableOp�
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype02#
!FusedBatchNormV3/ReadVariableOp_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:*
is_training( 2
FusedBatchNormV3S
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *�p}?2
Const�
IdentityIdentityFusedBatchNormV3:y:0 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�$
�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_22145

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_22130
assignmovingavg_1_22137
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
: *
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
: *
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+��������������������������� : : : : :*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/22130*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/22130*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_22130*
_output_shapes
: *
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/22130*
_output_shapes
: 2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/22130*
_output_shapes
: 2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_22130AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/22130*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/22137*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/22137*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_22137*
_output_shapes
: *
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/22137*
_output_shapes
: 2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/22137*
_output_shapes
: 2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_22137AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/22137*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
c
G__inference_activation_2_layer_call_and_return_conditional_losses_26361

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������@2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������@:& "
 
_user_specified_nameinputs
�
�
5__inference_batch_normalization_1_layer_call_fn_26336

inputs"
statefulpartitionedcall_args_1"
statefulpartitionedcall_args_2"
statefulpartitionedcall_args_3"
statefulpartitionedcall_args_4
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsstatefulpartitionedcall_args_1statefulpartitionedcall_args_2statefulpartitionedcall_args_3statefulpartitionedcall_args_4*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*A
_output_shapes/
-:+��������������������������� */
config_proto

GPU

CPU2 *0J 8*Y
fTRR
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_221762
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+��������������������������� 2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+��������������������������� ::::22
StatefulPartitionedCallStatefulPartitionedCall:& "
 
_user_specified_nameinputs
�#
�
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_24041

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_24026
assignmovingavg_1_24033
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/24026*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/24026*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_24026*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/24026*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/24026*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_24026AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/24026*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/24033*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/24033*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_24033*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/24033*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/24033*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_24033AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/24033*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�#
�
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_23740

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_23725
assignmovingavg_1_23732
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*K
_output_shapes9
7:���������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/23725*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/23725*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_23725*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/23725*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/23725*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_23725AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/23725*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/23732*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/23732*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_23732*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/23732*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/23732*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_23732AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/23732*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*/
_output_shapes
:���������@2

Identity"
identityIdentity:output:0*>
_input_shapes-
+:���������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs
�
a
E__inference_activation_layer_call_and_return_conditional_losses_23285

inputs
identityV
ReluReluinputs*
T0*/
_output_shapes
:���������00 2
Relun
IdentityIdentityRelu:activations:0*
T0*/
_output_shapes
:���������00 2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������00 :& "
 
_user_specified_nameinputs
�
h
>__inference_add_layer_call_and_return_conditional_losses_23473

inputs
inputs_1
identity_
addAddV2inputsinputs_1*
T0*/
_output_shapes
:���������00 2
addc
IdentityIdentityadd:z:0*
T0*/
_output_shapes
:���������00 2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:���������00 :���������00 :& "
 
_user_specified_nameinputs:&"
 
_user_specified_nameinputs
�$
�
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_27148

inputs
readvariableop_resource
readvariableop_1_resource
assignmovingavg_27133
assignmovingavg_1_27140
identity��#AssignMovingAvg/AssignSubVariableOp�AssignMovingAvg/ReadVariableOp�%AssignMovingAvg_1/AssignSubVariableOp� AssignMovingAvg_1/ReadVariableOp�ReadVariableOp�ReadVariableOp_1^
LogicalAnd/xConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/x^
LogicalAnd/yConst*
_output_shapes
: *
dtype0
*
value	B
 Z2
LogicalAnd/yl

LogicalAnd
LogicalAndLogicalAnd/x:output:0LogicalAnd/y:output:0*
_output_shapes
: 2

LogicalAndt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype02
ReadVariableOpz
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype02
ReadVariableOp_1Q
ConstConst*
_output_shapes
: *
dtype0*
valueB 2
ConstU
Const_1Const*
_output_shapes
: *
dtype0*
valueB 2	
Const_1�
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0Const:output:0Const_1:output:0*
T0*
U0*]
_output_shapesK
I:+���������������������������@:@:@:@:@:*
epsilon%o�:2
FusedBatchNormV3W
Const_2Const*
_output_shapes
: *
dtype0*
valueB
 *�p}?2	
Const_2�
AssignMovingAvg/sub/xConst*(
_class
loc:@AssignMovingAvg/27133*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg/sub/x�
AssignMovingAvg/subSubAssignMovingAvg/sub/x:output:0Const_2:output:0*
T0*(
_class
loc:@AssignMovingAvg/27133*
_output_shapes
: 2
AssignMovingAvg/sub�
AssignMovingAvg/ReadVariableOpReadVariableOpassignmovingavg_27133*
_output_shapes
:@*
dtype02 
AssignMovingAvg/ReadVariableOp�
AssignMovingAvg/sub_1Sub&AssignMovingAvg/ReadVariableOp:value:0FusedBatchNormV3:batch_mean:0*
T0*(
_class
loc:@AssignMovingAvg/27133*
_output_shapes
:@2
AssignMovingAvg/sub_1�
AssignMovingAvg/mulMulAssignMovingAvg/sub_1:z:0AssignMovingAvg/sub:z:0*
T0*(
_class
loc:@AssignMovingAvg/27133*
_output_shapes
:@2
AssignMovingAvg/mul�
#AssignMovingAvg/AssignSubVariableOpAssignSubVariableOpassignmovingavg_27133AssignMovingAvg/mul:z:0^AssignMovingAvg/ReadVariableOp*(
_class
loc:@AssignMovingAvg/27133*
_output_shapes
 *
dtype02%
#AssignMovingAvg/AssignSubVariableOp�
AssignMovingAvg_1/sub/xConst**
_class 
loc:@AssignMovingAvg_1/27140*
_output_shapes
: *
dtype0*
valueB
 *  �?2
AssignMovingAvg_1/sub/x�
AssignMovingAvg_1/subSub AssignMovingAvg_1/sub/x:output:0Const_2:output:0*
T0**
_class 
loc:@AssignMovingAvg_1/27140*
_output_shapes
: 2
AssignMovingAvg_1/sub�
 AssignMovingAvg_1/ReadVariableOpReadVariableOpassignmovingavg_1_27140*
_output_shapes
:@*
dtype02"
 AssignMovingAvg_1/ReadVariableOp�
AssignMovingAvg_1/sub_1Sub(AssignMovingAvg_1/ReadVariableOp:value:0!FusedBatchNormV3:batch_variance:0*
T0**
_class 
loc:@AssignMovingAvg_1/27140*
_output_shapes
:@2
AssignMovingAvg_1/sub_1�
AssignMovingAvg_1/mulMulAssignMovingAvg_1/sub_1:z:0AssignMovingAvg_1/sub:z:0*
T0**
_class 
loc:@AssignMovingAvg_1/27140*
_output_shapes
:@2
AssignMovingAvg_1/mul�
%AssignMovingAvg_1/AssignSubVariableOpAssignSubVariableOpassignmovingavg_1_27140AssignMovingAvg_1/mul:z:0!^AssignMovingAvg_1/ReadVariableOp**
_class 
loc:@AssignMovingAvg_1/27140*
_output_shapes
 *
dtype02'
%AssignMovingAvg_1/AssignSubVariableOp�
IdentityIdentityFusedBatchNormV3:y:0$^AssignMovingAvg/AssignSubVariableOp^AssignMovingAvg/ReadVariableOp&^AssignMovingAvg_1/AssignSubVariableOp!^AssignMovingAvg_1/ReadVariableOp^ReadVariableOp^ReadVariableOp_1*
T0*A
_output_shapes/
-:+���������������������������@2

Identity"
identityIdentity:output:0*P
_input_shapes?
=:+���������������������������@::::2J
#AssignMovingAvg/AssignSubVariableOp#AssignMovingAvg/AssignSubVariableOp2@
AssignMovingAvg/ReadVariableOpAssignMovingAvg/ReadVariableOp2N
%AssignMovingAvg_1/AssignSubVariableOp%AssignMovingAvg_1/AssignSubVariableOp2D
 AssignMovingAvg_1/ReadVariableOp AssignMovingAvg_1/ReadVariableOp2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:& "
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
C
board:
serving_default_board:0���������``F

repr_board8
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
��
layer-0
layer-1
layer-2
layer-3
layer_with_weights-0
layer-4
layer_with_weights-1
layer-5
layer-6
layer_with_weights-2
layer-7
	layer_with_weights-3
	layer-8

layer-9
layer_with_weights-4
layer-10
layer-11
layer_with_weights-5
layer-12
layer_with_weights-6
layer-13
layer-14
layer_with_weights-7
layer-15
layer_with_weights-8
layer-16
layer-17
layer_with_weights-9
layer-18
layer-19
layer-20
layer_with_weights-10
layer-21
layer-22
layer_with_weights-11
layer-23
layer_with_weights-12
layer-24
layer-25
layer_with_weights-13
layer-26
layer-27
layer-28
layer_with_weights-14
layer-29
layer-30
 layer_with_weights-15
 layer-31
!layer_with_weights-16
!layer-32
"layer-33
#layer_with_weights-17
#layer-34
$layer-35
%layer_with_weights-18
%layer-36
&trainable_variables
'regularization_losses
(	variables
)	keras_api
*
signatures
�__call__
+�&call_and_return_all_conditional_losses
�_default_save_signature"��
_tf_keras_model�{"class_name": "Model", "name": "Representation", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "Representation", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 1, 96, 96, 3], "dtype": "float32", "sparse": false, "ragged": false, "name": "board"}, "name": "board", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": [1, 96, 96, 3]}, "name": "reshape", "inbound_nodes": [[["board", 0, 0, {}]]]}, {"class_name": "Permute", "config": {"name": "permute", "trainable": true, "dtype": "float32", "dims": [2, 3, 1, 4]}, "name": "permute", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": [96, 96, 3]}, "name": "reshape_1", "inbound_nodes": [[["permute", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["reshape_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "res_conv_A_repr_a", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "res_conv_A_repr_a", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["res_conv_A_repr_a", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "res_conv_B_repr_a", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "res_conv_B_repr_a", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["res_conv_B_repr_a", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}], ["conv2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "res_conv_A_repr_b", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "res_conv_A_repr_b", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["res_conv_A_repr_b", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "res_conv_B_repr_b", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "res_conv_B_repr_b", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["res_conv_B_repr_b", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}], ["conv2d_1", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "average_pooling2d", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "res_conv_A_repr_c", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "res_conv_A_repr_c", "inbound_nodes": [[["average_pooling2d", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_4", "inbound_nodes": [[["res_conv_A_repr_c", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["activation_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "res_conv_B_repr_c", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "res_conv_B_repr_c", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_5", "inbound_nodes": [[["res_conv_B_repr_c", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["activation_5", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "name": "add_2", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}], ["average_pooling2d", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "average_pooling2d_1", "inbound_nodes": [[["add_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "res_conv_A_repr_d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "res_conv_A_repr_d", "inbound_nodes": [[["average_pooling2d_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_6", "inbound_nodes": [[["res_conv_A_repr_d", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["activation_6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "res_conv_B_repr_d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "res_conv_B_repr_d", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_7", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_7", "inbound_nodes": [[["res_conv_B_repr_d", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["activation_7", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "name": "add_3", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}], ["average_pooling2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "repr_board", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "repr_board", "inbound_nodes": [[["add_3", 0, 0, {}]]]}], "input_layers": [["board", 0, 0]], "output_layers": [["repr_board", 0, 0]]}, "is_graph_network": true, "keras_version": "2.2.4-tf", "backend": "tensorflow", "model_config": {"class_name": "Model", "config": {"name": "Representation", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": [null, 1, 96, 96, 3], "dtype": "float32", "sparse": false, "ragged": false, "name": "board"}, "name": "board", "inbound_nodes": []}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": [1, 96, 96, 3]}, "name": "reshape", "inbound_nodes": [[["board", 0, 0, {}]]]}, {"class_name": "Permute", "config": {"name": "permute", "trainable": true, "dtype": "float32", "dims": [2, 3, 1, 4]}, "name": "permute", "inbound_nodes": [[["reshape", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": [96, 96, 3]}, "name": "reshape_1", "inbound_nodes": [[["permute", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["reshape_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "res_conv_A_repr_a", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "res_conv_A_repr_a", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation", "inbound_nodes": [[["res_conv_A_repr_a", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization", "inbound_nodes": [[["activation", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "res_conv_B_repr_a", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "res_conv_B_repr_a", "inbound_nodes": [[["batch_normalization", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_1", "inbound_nodes": [[["res_conv_B_repr_a", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_1", "inbound_nodes": [[["activation_1", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["batch_normalization_1", 0, 0, {}], ["conv2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "res_conv_A_repr_b", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "res_conv_A_repr_b", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_2", "inbound_nodes": [[["res_conv_A_repr_b", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_2", "inbound_nodes": [[["activation_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "res_conv_B_repr_b", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "res_conv_B_repr_b", "inbound_nodes": [[["batch_normalization_2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_3", "inbound_nodes": [[["res_conv_B_repr_b", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_3", "inbound_nodes": [[["activation_3", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_1", "trainable": true, "dtype": "float32"}, "name": "add_1", "inbound_nodes": [[["batch_normalization_3", 0, 0, {}], ["conv2d_1", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "average_pooling2d", "inbound_nodes": [[["add_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "res_conv_A_repr_c", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "res_conv_A_repr_c", "inbound_nodes": [[["average_pooling2d", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_4", "inbound_nodes": [[["res_conv_A_repr_c", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_4", "inbound_nodes": [[["activation_4", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "res_conv_B_repr_c", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "res_conv_B_repr_c", "inbound_nodes": [[["batch_normalization_4", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_5", "inbound_nodes": [[["res_conv_B_repr_c", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_5", "inbound_nodes": [[["activation_5", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_2", "trainable": true, "dtype": "float32"}, "name": "add_2", "inbound_nodes": [[["batch_normalization_5", 0, 0, {}], ["average_pooling2d", 0, 0, {}]]]}, {"class_name": "AveragePooling2D", "config": {"name": "average_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "name": "average_pooling2d_1", "inbound_nodes": [[["add_2", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "res_conv_A_repr_d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "res_conv_A_repr_d", "inbound_nodes": [[["average_pooling2d_1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_6", "inbound_nodes": [[["res_conv_A_repr_d", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_6", "inbound_nodes": [[["activation_6", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "res_conv_B_repr_d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "res_conv_B_repr_d", "inbound_nodes": [[["batch_normalization_6", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "activation_7", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "activation_7", "inbound_nodes": [[["res_conv_B_repr_d", 0, 0, {}]]]}, {"class_name": "BatchNormalization", "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "name": "batch_normalization_7", "inbound_nodes": [[["activation_7", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add_3", "trainable": true, "dtype": "float32"}, "name": "add_3", "inbound_nodes": [[["batch_normalization_7", 0, 0, {}], ["average_pooling2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "repr_board", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "repr_board", "inbound_nodes": [[["add_3", 0, 0, {}]]]}], "input_layers": [["board", 0, 0]], "output_layers": [["repr_board", 0, 0]]}}}
�"�
_tf_keras_input_layer�{"class_name": "InputLayer", "name": "board", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": [null, 1, 96, 96, 3], "config": {"batch_input_shape": [null, 1, 96, 96, 3], "dtype": "float32", "sparse": false, "ragged": false, "name": "board"}}
�
+trainable_variables
,regularization_losses
-	variables
.	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": [1, 96, 96, 3]}}
�
/trainable_variables
0regularization_losses
1	variables
2	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Permute", "name": "permute", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "permute", "trainable": true, "dtype": "float32", "dims": [2, 3, 1, 4]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 5, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
3trainable_variables
4regularization_losses
5	variables
6	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Reshape", "name": "reshape_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": [96, 96, 3]}}
�

7kernel
8bias
9trainable_variables
:regularization_losses
;	variables
<	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 3}}}}
�

=kernel
>bias
?trainable_variables
@regularization_losses
A	variables
B	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "res_conv_A_repr_a", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "res_conv_A_repr_a", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
�
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation", "trainable": true, "dtype": "float32", "activation": "relu"}}
�
Gaxis
	Hgamma
Ibeta
Jmoving_mean
Kmoving_variance
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}}
�

Pkernel
Qbias
Rtrainable_variables
Sregularization_losses
T	variables
U	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "res_conv_B_repr_a", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "res_conv_B_repr_a", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
�
Vtrainable_variables
Wregularization_losses
X	variables
Y	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_1", "trainable": true, "dtype": "float32", "activation": "relu"}}
�
Zaxis
	[gamma
\beta
]moving_mean
^moving_variance
_trainable_variables
`regularization_losses
a	variables
b	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_1", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 32}}}}
�
ctrainable_variables
dregularization_losses
e	variables
f	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Add", "name": "add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add", "trainable": true, "dtype": "float32"}}
�

gkernel
hbias
itrainable_variables
jregularization_losses
k	variables
l	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [2, 2], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}}
�

mkernel
nbias
otrainable_variables
pregularization_losses
q	variables
r	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "res_conv_A_repr_b", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "res_conv_A_repr_b", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
�
strainable_variables
tregularization_losses
u	variables
v	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_2", "trainable": true, "dtype": "float32", "activation": "relu"}}
�
waxis
	xgamma
ybeta
zmoving_mean
{moving_variance
|trainable_variables
}regularization_losses
~	variables
	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_2", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_2", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
�
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "res_conv_B_repr_b", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "res_conv_B_repr_b", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_3", "trainable": true, "dtype": "float32", "activation": "relu"}}
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_3", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Add", "name": "add_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add_1", "trainable": true, "dtype": "float32"}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "AveragePooling2D", "name": "average_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "average_pooling2d", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "res_conv_A_repr_c", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "res_conv_A_repr_c", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_4", "trainable": true, "dtype": "float32", "activation": "relu"}}
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_4", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
�
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "res_conv_B_repr_c", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "res_conv_B_repr_c", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_5", "trainable": true, "dtype": "float32", "activation": "relu"}}
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_5", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_5", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Add", "name": "add_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add_2", "trainable": true, "dtype": "float32"}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "AveragePooling2D", "name": "average_pooling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "average_pooling2d_1", "trainable": true, "dtype": "float32", "pool_size": [2, 2], "padding": "valid", "strides": [2, 2], "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
�
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "res_conv_A_repr_d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "res_conv_A_repr_d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_6", "trainable": true, "dtype": "float32", "activation": "relu"}}
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_6", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_6", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
�
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "res_conv_B_repr_d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "res_conv_B_repr_d", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Activation", "name": "activation_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "activation_7", "trainable": true, "dtype": "float32", "activation": "relu"}}
�
	�axis

�gamma
	�beta
�moving_mean
�moving_variance
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "BatchNormalization", "name": "batch_normalization_7", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "config": {"name": "batch_normalization_7", "trainable": true, "dtype": "float32", "axis": [3], "momentum": 0.99, "epsilon": 0.001, "center": true, "scale": true, "beta_initializer": {"class_name": "Zeros", "config": {}}, "gamma_initializer": {"class_name": "Ones", "config": {}}, "moving_mean_initializer": {"class_name": "Zeros", "config": {}}, "moving_variance_initializer": {"class_name": "Ones", "config": {}}, "beta_regularizer": null, "gamma_regularizer": null, "beta_constraint": null, "gamma_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"3": 64}}}}
�
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Add", "name": "add_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "add_3", "trainable": true, "dtype": "float32"}}
�	
�kernel
	�bias
�trainable_variables
�regularization_losses
�	variables
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Conv2D", "name": "repr_board", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "repr_board", "trainable": true, "dtype": "float32", "filters": 16, "kernel_size": [3, 3], "strides": [1, 1], "padding": "same", "data_format": "channels_last", "dilation_rate": [1, 1], "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "bias_regularizer": {"class_name": "L1L2", "config": {"l1": 0.0, "l2": 9.999999747378752e-06}}, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}}
�
70
81
=2
>3
H4
I5
P6
Q7
[8
\9
g10
h11
m12
n13
x14
y15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37"
trackable_list_wrapper
p
�0
�1
�2
�3
�4
�5
�6
�7
�8
�9"
trackable_list_wrapper
�
70
81
=2
>3
H4
I5
J6
K7
P8
Q9
[10
\11
]12
^13
g14
h15
m16
n17
x18
y19
z20
{21
�22
�23
�24
�25
�26
�27
�28
�29
�30
�31
�32
�33
�34
�35
�36
�37
�38
�39
�40
�41
�42
�43
�44
�45
�46
�47
�48
�49
�50
�51
�52
�53"
trackable_list_wrapper
�
�non_trainable_variables
�layers
&trainable_variables
 �layer_regularization_losses
�metrics
'regularization_losses
(	variables
�__call__
�_default_save_signature
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
+trainable_variables
 �layer_regularization_losses
�metrics
,regularization_losses
-	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
/trainable_variables
 �layer_regularization_losses
�metrics
0regularization_losses
1	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
3trainable_variables
 �layer_regularization_losses
�metrics
4regularization_losses
5	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
':% 2conv2d/kernel
: 2conv2d/bias
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
�
�non_trainable_variables
�layers
9trainable_variables
 �layer_regularization_losses
�metrics
:regularization_losses
;	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
2:0  2res_conv_A_repr_a/kernel
$:" 2res_conv_A_repr_a/bias
.
=0
>1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
?trainable_variables
 �layer_regularization_losses
�metrics
@regularization_losses
A	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
Ctrainable_variables
 �layer_regularization_losses
�metrics
Dregularization_losses
E	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
':% 2batch_normalization/gamma
&:$ 2batch_normalization/beta
/:-  (2batch_normalization/moving_mean
3:1  (2#batch_normalization/moving_variance
.
H0
I1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
H0
I1
J2
K3"
trackable_list_wrapper
�
�non_trainable_variables
�layers
Ltrainable_variables
 �layer_regularization_losses
�metrics
Mregularization_losses
N	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
2:0  2res_conv_B_repr_a/kernel
$:" 2res_conv_B_repr_a/bias
.
P0
Q1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
Rtrainable_variables
 �layer_regularization_losses
�metrics
Sregularization_losses
T	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
Vtrainable_variables
 �layer_regularization_losses
�metrics
Wregularization_losses
X	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):' 2batch_normalization_1/gamma
(:& 2batch_normalization_1/beta
1:/  (2!batch_normalization_1/moving_mean
5:3  (2%batch_normalization_1/moving_variance
.
[0
\1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
[0
\1
]2
^3"
trackable_list_wrapper
�
�non_trainable_variables
�layers
_trainable_variables
 �layer_regularization_losses
�metrics
`regularization_losses
a	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
ctrainable_variables
 �layer_regularization_losses
�metrics
dregularization_losses
e	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
):' @2conv2d_1/kernel
:@2conv2d_1/bias
.
g0
h1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
g0
h1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
itrainable_variables
 �layer_regularization_losses
�metrics
jregularization_losses
k	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
2:0@@2res_conv_A_repr_b/kernel
$:"@2res_conv_A_repr_b/bias
.
m0
n1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
.
m0
n1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
otrainable_variables
 �layer_regularization_losses
�metrics
pregularization_losses
q	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
strainable_variables
 �layer_regularization_losses
�metrics
tregularization_losses
u	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_2/gamma
(:&@2batch_normalization_2/beta
1:/@ (2!batch_normalization_2/moving_mean
5:3@ (2%batch_normalization_2/moving_variance
.
x0
y1"
trackable_list_wrapper
 "
trackable_list_wrapper
<
x0
y1
z2
{3"
trackable_list_wrapper
�
�non_trainable_variables
�layers
|trainable_variables
 �layer_regularization_losses
�metrics
}regularization_losses
~	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
2:0@@2res_conv_B_repr_b/kernel
$:"@2res_conv_B_repr_b/bias
0
�0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_3/gamma
(:&@2batch_normalization_3/beta
1:/@ (2!batch_normalization_3/moving_mean
5:3@ (2%batch_normalization_3/moving_variance
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
2:0@@2res_conv_A_repr_c/kernel
$:"@2res_conv_A_repr_c/bias
0
�0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_4/gamma
(:&@2batch_normalization_4/beta
1:/@ (2!batch_normalization_4/moving_mean
5:3@ (2%batch_normalization_4/moving_variance
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
2:0@@2res_conv_B_repr_c/kernel
$:"@2res_conv_B_repr_c/bias
0
�0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_5/gamma
(:&@2batch_normalization_5/beta
1:/@ (2!batch_normalization_5/moving_mean
5:3@ (2%batch_normalization_5/moving_variance
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
2:0@@2res_conv_A_repr_d/kernel
$:"@2res_conv_A_repr_d/bias
0
�0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_6/gamma
(:&@2batch_normalization_6/beta
1:/@ (2!batch_normalization_6/moving_mean
5:3@ (2%batch_normalization_6/moving_variance
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
2:0@@2res_conv_B_repr_d/kernel
$:"@2res_conv_B_repr_d/bias
0
�0
�1"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
):'@2batch_normalization_7/gamma
(:&@2batch_normalization_7/beta
1:/@ (2!batch_normalization_7/moving_mean
5:3@ (2%batch_normalization_7/moving_variance
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
+:)@2repr_board/kernel
:2repr_board/bias
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
�
�non_trainable_variables
�layers
�trainable_variables
 �layer_regularization_losses
�metrics
�regularization_losses
�	variables
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
J0
K1
]2
^3
z4
{5
�6
�7
�8
�9
�10
�11
�12
�13
�14
�15"
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
 31
!32
"33
#34
$35
%36"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
]0
^1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
z0
{1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�2�
.__inference_Representation_layer_call_fn_24594
.__inference_Representation_layer_call_fn_24827
.__inference_Representation_layer_call_fn_25882
.__inference_Representation_layer_call_fn_25941�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
I__inference_Representation_layer_call_and_return_conditional_losses_24360
I__inference_Representation_layer_call_and_return_conditional_losses_24186
I__inference_Representation_layer_call_and_return_conditional_losses_25823
I__inference_Representation_layer_call_and_return_conditional_losses_25491�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
 __inference__wrapped_model_21830�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *0�-
+�(
board���������``
�2�
'__inference_reshape_layer_call_fn_25961�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
B__inference_reshape_layer_call_and_return_conditional_losses_25956�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
'__inference_permute_layer_call_fn_21843�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *M�J
H�EA���������������������������������������������
�2�
B__inference_permute_layer_call_and_return_conditional_losses_21837�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *M�J
H�EA���������������������������������������������
�2�
)__inference_reshape_1_layer_call_fn_25980�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
D__inference_reshape_1_layer_call_and_return_conditional_losses_25975�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
&__inference_conv2d_layer_call_fn_21863�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
A__inference_conv2d_layer_call_and_return_conditional_losses_21855�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������
�2�
1__inference_res_conv_A_repr_a_layer_call_fn_21891�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
L__inference_res_conv_A_repr_a_layer_call_and_return_conditional_losses_21883�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
*__inference_activation_layer_call_fn_25998�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_activation_layer_call_and_return_conditional_losses_25993�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
3__inference_batch_normalization_layer_call_fn_26075
3__inference_batch_normalization_layer_call_fn_26158
3__inference_batch_normalization_layer_call_fn_26084
3__inference_batch_normalization_layer_call_fn_26149�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_26044
N__inference_batch_normalization_layer_call_and_return_conditional_losses_26118
N__inference_batch_normalization_layer_call_and_return_conditional_losses_26066
N__inference_batch_normalization_layer_call_and_return_conditional_losses_26140�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
1__inference_res_conv_B_repr_a_layer_call_fn_22051�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
L__inference_res_conv_B_repr_a_layer_call_and_return_conditional_losses_22043�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
,__inference_activation_1_layer_call_fn_26176�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_activation_1_layer_call_and_return_conditional_losses_26171�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
5__inference_batch_normalization_1_layer_call_fn_26253
5__inference_batch_normalization_1_layer_call_fn_26262
5__inference_batch_normalization_1_layer_call_fn_26327
5__inference_batch_normalization_1_layer_call_fn_26336�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_26318
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_26296
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_26222
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_26244�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
#__inference_add_layer_call_fn_26348�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
>__inference_add_layer_call_and_return_conditional_losses_26342�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_conv2d_1_layer_call_fn_22203�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
C__inference_conv2d_1_layer_call_and_return_conditional_losses_22195�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+��������������������������� 
�2�
1__inference_res_conv_A_repr_b_layer_call_fn_22231�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
L__inference_res_conv_A_repr_b_layer_call_and_return_conditional_losses_22223�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
,__inference_activation_2_layer_call_fn_26366�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_activation_2_layer_call_and_return_conditional_losses_26361�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
5__inference_batch_normalization_2_layer_call_fn_26526
5__inference_batch_normalization_2_layer_call_fn_26443
5__inference_batch_normalization_2_layer_call_fn_26452
5__inference_batch_normalization_2_layer_call_fn_26517�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26412
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26434
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26486
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26508�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
1__inference_res_conv_B_repr_b_layer_call_fn_22391�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
L__inference_res_conv_B_repr_b_layer_call_and_return_conditional_losses_22383�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
,__inference_activation_3_layer_call_fn_26544�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_activation_3_layer_call_and_return_conditional_losses_26539�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
5__inference_batch_normalization_3_layer_call_fn_26621
5__inference_batch_normalization_3_layer_call_fn_26695
5__inference_batch_normalization_3_layer_call_fn_26704
5__inference_batch_normalization_3_layer_call_fn_26630�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_26686
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_26590
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_26612
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_26664�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
%__inference_add_1_layer_call_fn_26716�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_add_1_layer_call_and_return_conditional_losses_26710�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
1__inference_average_pooling2d_layer_call_fn_22535�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_22529�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
1__inference_res_conv_A_repr_c_layer_call_fn_22563�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
L__inference_res_conv_A_repr_c_layer_call_and_return_conditional_losses_22555�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
,__inference_activation_4_layer_call_fn_26734�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_activation_4_layer_call_and_return_conditional_losses_26729�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
5__inference_batch_normalization_4_layer_call_fn_26820
5__inference_batch_normalization_4_layer_call_fn_26894
5__inference_batch_normalization_4_layer_call_fn_26885
5__inference_batch_normalization_4_layer_call_fn_26811�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26876
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26854
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26802
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26780�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
1__inference_res_conv_B_repr_c_layer_call_fn_22723�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
L__inference_res_conv_B_repr_c_layer_call_and_return_conditional_losses_22715�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
,__inference_activation_5_layer_call_fn_26912�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_activation_5_layer_call_and_return_conditional_losses_26907�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
5__inference_batch_normalization_5_layer_call_fn_26998
5__inference_batch_normalization_5_layer_call_fn_26989
5__inference_batch_normalization_5_layer_call_fn_27063
5__inference_batch_normalization_5_layer_call_fn_27072�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_26980
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_27032
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_27054
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_26958�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
%__inference_add_2_layer_call_fn_27084�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_add_2_layer_call_and_return_conditional_losses_27078�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
3__inference_average_pooling2d_1_layer_call_fn_22867�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
N__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_22861�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *@�=
;�84������������������������������������
�2�
1__inference_res_conv_A_repr_d_layer_call_fn_22895�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
L__inference_res_conv_A_repr_d_layer_call_and_return_conditional_losses_22887�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
,__inference_activation_6_layer_call_fn_27102�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_activation_6_layer_call_and_return_conditional_losses_27097�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
5__inference_batch_normalization_6_layer_call_fn_27253
5__inference_batch_normalization_6_layer_call_fn_27262
5__inference_batch_normalization_6_layer_call_fn_27188
5__inference_batch_normalization_6_layer_call_fn_27179�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_27148
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_27244
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_27222
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_27170�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
1__inference_res_conv_B_repr_d_layer_call_fn_23055�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
L__inference_res_conv_B_repr_d_layer_call_and_return_conditional_losses_23047�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
,__inference_activation_7_layer_call_fn_27280�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
G__inference_activation_7_layer_call_and_return_conditional_losses_27275�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
5__inference_batch_normalization_7_layer_call_fn_27357
5__inference_batch_normalization_7_layer_call_fn_27431
5__inference_batch_normalization_7_layer_call_fn_27366
5__inference_batch_normalization_7_layer_call_fn_27440�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_27348
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_27400
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_27326
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_27422�
���
FullArgSpec)
args!�
jself
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
%__inference_add_3_layer_call_fn_27452�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
@__inference_add_3_layer_call_and_return_conditional_losses_27446�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_repr_board_layer_call_fn_23224�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
E__inference_repr_board_layer_call_and_return_conditional_losses_23216�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *7�4
2�/+���������������������������@
�2�
__inference_loss_fn_0_27481�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_1_27494�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_2_27507�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_3_27520�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_4_27533�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_5_27546�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_6_27559�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_7_27572�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_8_27585�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_9_27598�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
0B.
#__inference_signature_wrapper_25063board�
I__inference_Representation_layer_call_and_return_conditional_losses_24186�V78=>HIJKPQ[\]^ghmnxyz{��������������������������������B�?
8�5
+�(
board���������``
p

 
� "-�*
#� 
0���������
� �
I__inference_Representation_layer_call_and_return_conditional_losses_24360�V78=>HIJKPQ[\]^ghmnxyz{��������������������������������B�?
8�5
+�(
board���������``
p 

 
� "-�*
#� 
0���������
� �
I__inference_Representation_layer_call_and_return_conditional_losses_25491�V78=>HIJKPQ[\]^ghmnxyz{��������������������������������C�@
9�6
,�)
inputs���������``
p

 
� "-�*
#� 
0���������
� �
I__inference_Representation_layer_call_and_return_conditional_losses_25823�V78=>HIJKPQ[\]^ghmnxyz{��������������������������������C�@
9�6
,�)
inputs���������``
p 

 
� "-�*
#� 
0���������
� �
.__inference_Representation_layer_call_fn_24594�V78=>HIJKPQ[\]^ghmnxyz{��������������������������������B�?
8�5
+�(
board���������``
p

 
� " �����������
.__inference_Representation_layer_call_fn_24827�V78=>HIJKPQ[\]^ghmnxyz{��������������������������������B�?
8�5
+�(
board���������``
p 

 
� " �����������
.__inference_Representation_layer_call_fn_25882�V78=>HIJKPQ[\]^ghmnxyz{��������������������������������C�@
9�6
,�)
inputs���������``
p

 
� " �����������
.__inference_Representation_layer_call_fn_25941�V78=>HIJKPQ[\]^ghmnxyz{��������������������������������C�@
9�6
,�)
inputs���������``
p 

 
� " �����������
 __inference__wrapped_model_21830�V78=>HIJKPQ[\]^ghmnxyz{��������������������������������:�7
0�-
+�(
board���������``
� "?�<
:

repr_board,�)

repr_board����������
G__inference_activation_1_layer_call_and_return_conditional_losses_26171h7�4
-�*
(�%
inputs���������00 
� "-�*
#� 
0���������00 
� �
,__inference_activation_1_layer_call_fn_26176[7�4
-�*
(�%
inputs���������00 
� " ����������00 �
G__inference_activation_2_layer_call_and_return_conditional_losses_26361h7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
,__inference_activation_2_layer_call_fn_26366[7�4
-�*
(�%
inputs���������@
� " ����������@�
G__inference_activation_3_layer_call_and_return_conditional_losses_26539h7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
,__inference_activation_3_layer_call_fn_26544[7�4
-�*
(�%
inputs���������@
� " ����������@�
G__inference_activation_4_layer_call_and_return_conditional_losses_26729h7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
,__inference_activation_4_layer_call_fn_26734[7�4
-�*
(�%
inputs���������@
� " ����������@�
G__inference_activation_5_layer_call_and_return_conditional_losses_26907h7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
,__inference_activation_5_layer_call_fn_26912[7�4
-�*
(�%
inputs���������@
� " ����������@�
G__inference_activation_6_layer_call_and_return_conditional_losses_27097h7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
,__inference_activation_6_layer_call_fn_27102[7�4
-�*
(�%
inputs���������@
� " ����������@�
G__inference_activation_7_layer_call_and_return_conditional_losses_27275h7�4
-�*
(�%
inputs���������@
� "-�*
#� 
0���������@
� �
,__inference_activation_7_layer_call_fn_27280[7�4
-�*
(�%
inputs���������@
� " ����������@�
E__inference_activation_layer_call_and_return_conditional_losses_25993h7�4
-�*
(�%
inputs���������00 
� "-�*
#� 
0���������00 
� �
*__inference_activation_layer_call_fn_25998[7�4
-�*
(�%
inputs���������00 
� " ����������00 �
@__inference_add_1_layer_call_and_return_conditional_losses_26710�j�g
`�]
[�X
*�'
inputs/0���������@
*�'
inputs/1���������@
� "-�*
#� 
0���������@
� �
%__inference_add_1_layer_call_fn_26716�j�g
`�]
[�X
*�'
inputs/0���������@
*�'
inputs/1���������@
� " ����������@�
@__inference_add_2_layer_call_and_return_conditional_losses_27078�j�g
`�]
[�X
*�'
inputs/0���������@
*�'
inputs/1���������@
� "-�*
#� 
0���������@
� �
%__inference_add_2_layer_call_fn_27084�j�g
`�]
[�X
*�'
inputs/0���������@
*�'
inputs/1���������@
� " ����������@�
@__inference_add_3_layer_call_and_return_conditional_losses_27446�j�g
`�]
[�X
*�'
inputs/0���������@
*�'
inputs/1���������@
� "-�*
#� 
0���������@
� �
%__inference_add_3_layer_call_fn_27452�j�g
`�]
[�X
*�'
inputs/0���������@
*�'
inputs/1���������@
� " ����������@�
>__inference_add_layer_call_and_return_conditional_losses_26342�j�g
`�]
[�X
*�'
inputs/0���������00 
*�'
inputs/1���������00 
� "-�*
#� 
0���������00 
� �
#__inference_add_layer_call_fn_26348�j�g
`�]
[�X
*�'
inputs/0���������00 
*�'
inputs/1���������00 
� " ����������00 �
N__inference_average_pooling2d_1_layer_call_and_return_conditional_losses_22861�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
3__inference_average_pooling2d_1_layer_call_fn_22867�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
L__inference_average_pooling2d_layer_call_and_return_conditional_losses_22529�R�O
H�E
C�@
inputs4������������������������������������
� "H�E
>�;
04������������������������������������
� �
1__inference_average_pooling2d_layer_call_fn_22535�R�O
H�E
C�@
inputs4������������������������������������
� ";�84�������������������������������������
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_26222r[\]^;�8
1�.
(�%
inputs���������00 
p
� "-�*
#� 
0���������00 
� �
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_26244r[\]^;�8
1�.
(�%
inputs���������00 
p 
� "-�*
#� 
0���������00 
� �
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_26296�[\]^M�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
P__inference_batch_normalization_1_layer_call_and_return_conditional_losses_26318�[\]^M�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
5__inference_batch_normalization_1_layer_call_fn_26253e[\]^;�8
1�.
(�%
inputs���������00 
p
� " ����������00 �
5__inference_batch_normalization_1_layer_call_fn_26262e[\]^;�8
1�.
(�%
inputs���������00 
p 
� " ����������00 �
5__inference_batch_normalization_1_layer_call_fn_26327�[\]^M�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
5__inference_batch_normalization_1_layer_call_fn_26336�[\]^M�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26412�xyz{M�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26434�xyz{M�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26486rxyz{;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
P__inference_batch_normalization_2_layer_call_and_return_conditional_losses_26508rxyz{;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
5__inference_batch_normalization_2_layer_call_fn_26443�xyz{M�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
5__inference_batch_normalization_2_layer_call_fn_26452�xyz{M�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
5__inference_batch_normalization_2_layer_call_fn_26517exyz{;�8
1�.
(�%
inputs���������@
p
� " ����������@�
5__inference_batch_normalization_2_layer_call_fn_26526exyz{;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_26590v����;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_26612v����;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_26664�����M�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
P__inference_batch_normalization_3_layer_call_and_return_conditional_losses_26686�����M�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
5__inference_batch_normalization_3_layer_call_fn_26621i����;�8
1�.
(�%
inputs���������@
p
� " ����������@�
5__inference_batch_normalization_3_layer_call_fn_26630i����;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
5__inference_batch_normalization_3_layer_call_fn_26695�����M�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
5__inference_batch_normalization_3_layer_call_fn_26704�����M�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26780�����M�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26802�����M�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26854v����;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
P__inference_batch_normalization_4_layer_call_and_return_conditional_losses_26876v����;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
5__inference_batch_normalization_4_layer_call_fn_26811�����M�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
5__inference_batch_normalization_4_layer_call_fn_26820�����M�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
5__inference_batch_normalization_4_layer_call_fn_26885i����;�8
1�.
(�%
inputs���������@
p
� " ����������@�
5__inference_batch_normalization_4_layer_call_fn_26894i����;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_26958v����;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_26980v����;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_27032�����M�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
P__inference_batch_normalization_5_layer_call_and_return_conditional_losses_27054�����M�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
5__inference_batch_normalization_5_layer_call_fn_26989i����;�8
1�.
(�%
inputs���������@
p
� " ����������@�
5__inference_batch_normalization_5_layer_call_fn_26998i����;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
5__inference_batch_normalization_5_layer_call_fn_27063�����M�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
5__inference_batch_normalization_5_layer_call_fn_27072�����M�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_27148�����M�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_27170�����M�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_27222v����;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
P__inference_batch_normalization_6_layer_call_and_return_conditional_losses_27244v����;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
5__inference_batch_normalization_6_layer_call_fn_27179�����M�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
5__inference_batch_normalization_6_layer_call_fn_27188�����M�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
5__inference_batch_normalization_6_layer_call_fn_27253i����;�8
1�.
(�%
inputs���������@
p
� " ����������@�
5__inference_batch_normalization_6_layer_call_fn_27262i����;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_27326�����M�J
C�@
:�7
inputs+���������������������������@
p
� "?�<
5�2
0+���������������������������@
� �
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_27348�����M�J
C�@
:�7
inputs+���������������������������@
p 
� "?�<
5�2
0+���������������������������@
� �
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_27400v����;�8
1�.
(�%
inputs���������@
p
� "-�*
#� 
0���������@
� �
P__inference_batch_normalization_7_layer_call_and_return_conditional_losses_27422v����;�8
1�.
(�%
inputs���������@
p 
� "-�*
#� 
0���������@
� �
5__inference_batch_normalization_7_layer_call_fn_27357�����M�J
C�@
:�7
inputs+���������������������������@
p
� "2�/+���������������������������@�
5__inference_batch_normalization_7_layer_call_fn_27366�����M�J
C�@
:�7
inputs+���������������������������@
p 
� "2�/+���������������������������@�
5__inference_batch_normalization_7_layer_call_fn_27431i����;�8
1�.
(�%
inputs���������@
p
� " ����������@�
5__inference_batch_normalization_7_layer_call_fn_27440i����;�8
1�.
(�%
inputs���������@
p 
� " ����������@�
N__inference_batch_normalization_layer_call_and_return_conditional_losses_26044rHIJK;�8
1�.
(�%
inputs���������00 
p
� "-�*
#� 
0���������00 
� �
N__inference_batch_normalization_layer_call_and_return_conditional_losses_26066rHIJK;�8
1�.
(�%
inputs���������00 
p 
� "-�*
#� 
0���������00 
� �
N__inference_batch_normalization_layer_call_and_return_conditional_losses_26118�HIJKM�J
C�@
:�7
inputs+��������������������������� 
p
� "?�<
5�2
0+��������������������������� 
� �
N__inference_batch_normalization_layer_call_and_return_conditional_losses_26140�HIJKM�J
C�@
:�7
inputs+��������������������������� 
p 
� "?�<
5�2
0+��������������������������� 
� �
3__inference_batch_normalization_layer_call_fn_26075eHIJK;�8
1�.
(�%
inputs���������00 
p
� " ����������00 �
3__inference_batch_normalization_layer_call_fn_26084eHIJK;�8
1�.
(�%
inputs���������00 
p 
� " ����������00 �
3__inference_batch_normalization_layer_call_fn_26149�HIJKM�J
C�@
:�7
inputs+��������������������������� 
p
� "2�/+��������������������������� �
3__inference_batch_normalization_layer_call_fn_26158�HIJKM�J
C�@
:�7
inputs+��������������������������� 
p 
� "2�/+��������������������������� �
C__inference_conv2d_1_layer_call_and_return_conditional_losses_22195�ghI�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+���������������������������@
� �
(__inference_conv2d_1_layer_call_fn_22203�ghI�F
?�<
:�7
inputs+��������������������������� 
� "2�/+���������������������������@�
A__inference_conv2d_layer_call_and_return_conditional_losses_21855�78I�F
?�<
:�7
inputs+���������������������������
� "?�<
5�2
0+��������������������������� 
� �
&__inference_conv2d_layer_call_fn_21863�78I�F
?�<
:�7
inputs+���������������������������
� "2�/+��������������������������� :
__inference_loss_fn_0_27481=�

� 
� "� :
__inference_loss_fn_1_27494P�

� 
� "� :
__inference_loss_fn_2_27507m�

� 
� "� ;
__inference_loss_fn_3_27520��

� 
� "� ;
__inference_loss_fn_4_27533��

� 
� "� ;
__inference_loss_fn_5_27546��

� 
� "� ;
__inference_loss_fn_6_27559��

� 
� "� ;
__inference_loss_fn_7_27572��

� 
� "� ;
__inference_loss_fn_8_27585��

� 
� "� ;
__inference_loss_fn_9_27598��

� 
� "� �
B__inference_permute_layer_call_and_return_conditional_losses_21837�_�\
U�R
P�M
inputsA���������������������������������������������
� "U�R
K�H
0A���������������������������������������������
� �
'__inference_permute_layer_call_fn_21843�_�\
U�R
P�M
inputsA���������������������������������������������
� "H�EA����������������������������������������������
E__inference_repr_board_layer_call_and_return_conditional_losses_23216���I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������
� �
*__inference_repr_board_layer_call_fn_23224���I�F
?�<
:�7
inputs+���������������������������@
� "2�/+����������������������������
L__inference_res_conv_A_repr_a_layer_call_and_return_conditional_losses_21883�=>I�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+��������������������������� 
� �
1__inference_res_conv_A_repr_a_layer_call_fn_21891�=>I�F
?�<
:�7
inputs+��������������������������� 
� "2�/+��������������������������� �
L__inference_res_conv_A_repr_b_layer_call_and_return_conditional_losses_22223�mnI�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������@
� �
1__inference_res_conv_A_repr_b_layer_call_fn_22231�mnI�F
?�<
:�7
inputs+���������������������������@
� "2�/+���������������������������@�
L__inference_res_conv_A_repr_c_layer_call_and_return_conditional_losses_22555���I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������@
� �
1__inference_res_conv_A_repr_c_layer_call_fn_22563���I�F
?�<
:�7
inputs+���������������������������@
� "2�/+���������������������������@�
L__inference_res_conv_A_repr_d_layer_call_and_return_conditional_losses_22887���I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������@
� �
1__inference_res_conv_A_repr_d_layer_call_fn_22895���I�F
?�<
:�7
inputs+���������������������������@
� "2�/+���������������������������@�
L__inference_res_conv_B_repr_a_layer_call_and_return_conditional_losses_22043�PQI�F
?�<
:�7
inputs+��������������������������� 
� "?�<
5�2
0+��������������������������� 
� �
1__inference_res_conv_B_repr_a_layer_call_fn_22051�PQI�F
?�<
:�7
inputs+��������������������������� 
� "2�/+��������������������������� �
L__inference_res_conv_B_repr_b_layer_call_and_return_conditional_losses_22383���I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������@
� �
1__inference_res_conv_B_repr_b_layer_call_fn_22391���I�F
?�<
:�7
inputs+���������������������������@
� "2�/+���������������������������@�
L__inference_res_conv_B_repr_c_layer_call_and_return_conditional_losses_22715���I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������@
� �
1__inference_res_conv_B_repr_c_layer_call_fn_22723���I�F
?�<
:�7
inputs+���������������������������@
� "2�/+���������������������������@�
L__inference_res_conv_B_repr_d_layer_call_and_return_conditional_losses_23047���I�F
?�<
:�7
inputs+���������������������������@
� "?�<
5�2
0+���������������������������@
� �
1__inference_res_conv_B_repr_d_layer_call_fn_23055���I�F
?�<
:�7
inputs+���������������������������@
� "2�/+���������������������������@�
D__inference_reshape_1_layer_call_and_return_conditional_losses_25975l;�8
1�.
,�)
inputs���������``
� "-�*
#� 
0���������``
� �
)__inference_reshape_1_layer_call_fn_25980_;�8
1�.
,�)
inputs���������``
� " ����������``�
B__inference_reshape_layer_call_and_return_conditional_losses_25956p;�8
1�.
,�)
inputs���������``
� "1�.
'�$
0���������``
� �
'__inference_reshape_layer_call_fn_25961c;�8
1�.
,�)
inputs���������``
� "$�!���������``�
#__inference_signature_wrapper_25063�V78=>HIJKPQ[\]^ghmnxyz{��������������������������������C�@
� 
9�6
4
board+�(
board���������``"?�<
:

repr_board,�)

repr_board���������