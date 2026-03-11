"""
`Learn the Basics <intro.html>`_ ||
`Quickstart <quickstart_tutorial.html>`_ ||
`Tensors <tensorqs_tutorial.html>`_ ||
`Datasets & DataLoaders <data_tutorial.html>`_ ||
`Transforms <transforms_tutorial.html>`_ ||
`Build Model <buildmodel_tutorial.html>`_ ||
**Autograd** ||
`Optimization <optimization_tutorial.html>`_ ||
`Save & Load Model <saveloadrun_tutorial.html>`_

使用 ``torch.autograd`` 进行自动微分
====================================

在训练神经网络时，最常用的算法是
**反向传播**。在该算法中，参数（模型权重）会根据
损失函数相对于给定参数的**梯度**进行调整。

为了计算这些梯度，PyTorch 内置了一个名为
``torch.autograd`` 的求导引擎。它支持对任意
计算图的梯度进行自动计算。

考虑一个最简单的单层神经网络，它包含输入 ``x``、
参数 ``w`` 和 ``b``，以及某个损失函数。它可以按如下方式
在 PyTorch 中定义：
"""

import torch

x = torch.ones(5)  # 输入张量
y = torch.zeros(3)  # 期望输出
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)


######################################################################
# 张量、函数与计算图
# ------------------
#
# 这段代码定义了如下**计算图**：
#
# .. figure:: /_static/img/basics/comp-graph.png
#    :alt:
#
# 在这个网络中，``w`` 和 ``b`` 是需要被优化的**参数**。
# 因此，我们需要能够计算损失函数相对于这些变量的梯度。
# 为了做到这一点，我们为这些张量设置 ``requires_grad`` 属性。

#######################################################################
# .. note:: 你可以在创建张量时设置 ``requires_grad`` 的值，
#           也可以之后通过 ``x.requires_grad_(True)`` 方法来设置。

#######################################################################
# 我们应用到张量上以构建计算图的函数，实际上是 ``Function`` 类的对象。
# 这个对象知道如何在*前向传播*中计算函数，也知道如何在
# *反向传播*步骤中计算它的导数。对反向传播函数的引用
# 保存在张量的 ``grad_fn`` 属性中。你可以在
# `文档 <https://pytorch.org/docs/stable/autograd.html#function>`__
# 中找到更多关于 ``Function`` 的信息。
#

print(f"Gradient function for z = {z.grad_fn}")
print(f"Gradient function for loss = {loss.grad_fn}")

######################################################################
# 梯度计算
# --------
#
# 为了优化神经网络中的参数权重，我们需要计算损失函数相对于参数的导数，
# 也就是在固定 ``x`` 和 ``y`` 的情况下，计算
# :math:`\frac{\partial loss}{\partial w}` 和
# :math:`\frac{\partial loss}{\partial b}`。要计算这些导数，
# 我们调用 ``loss.backward()``，然后从 ``w.grad`` 和 ``b.grad``
# 中取出对应的值：
#

loss.backward()
print(w.grad)
print(b.grad)


######################################################################
# .. note::
#   - 我们只能获取计算图中叶子节点的 ``grad`` 属性，这些叶子节点
#     需要将 ``requires_grad`` 属性设置为 ``True``。对于图中的
#     其他节点，梯度是不可用的。
#   - 出于性能原因，在同一个计算图上通常只能使用 ``backward``
#     进行一次梯度计算。如果需要在同一个图上多次调用 ``backward``，
#     则需要在调用时传入 ``retain_graph=True``。
#


######################################################################
# 禁用梯度跟踪
# ------------
#
# 默认情况下，所有设置了 ``requires_grad=True`` 的张量都会跟踪
# 它们的计算历史，并支持梯度计算。不过在某些情况下我们并不需要这样做，
# 例如模型已经训练完成，只想把它应用到一些输入数据上，也就是只进行
# 网络的*前向*计算。我们可以通过使用 ``torch.no_grad()`` 代码块
# 包裹计算过程来停止跟踪：
#

z = torch.matmul(x, w) + b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w) + b
print(z.requires_grad)


######################################################################
# 达到同样效果的另一种方式，是对张量使用 ``detach()`` 方法：
#

z = torch.matmul(x, w) + b
z_det = z.detach()
print(z_det.requires_grad)

######################################################################
# 你可能会出于以下原因禁用梯度跟踪：
#   - 将神经网络中的某些参数标记为**冻结参数**。
#   - 当你只进行前向传播时，为了**加快计算速度**，因为不跟踪
#     梯度的张量在计算时会更高效。


######################################################################

######################################################################
# 关于计算图的更多内容
# --------------------
# 从概念上讲，autograd 会将数据（张量）以及所有已执行的操作
# （连同新生成的张量）记录在一个由
# `Function <https://pytorch.org/docs/stable/autograd.html#torch.autograd.Function>`__
# 对象构成的有向无环图（DAG）中。在这个 DAG 里，叶子是输入张量，
# 根是输出张量。通过从根追踪到叶子，你可以利用链式法则自动计算梯度。
#
# 在前向传播过程中，autograd 会同时做两件事：
#
# - 运行请求的操作以计算结果张量
# - 在 DAG 中维护该操作的*梯度函数*
#
# 当在 DAG 的根节点上调用 ``.backward()`` 时，反向传播就会启动。
# 随后 ``autograd`` 会：
#
# - 从每个 ``.grad_fn`` 计算梯度
# - 将梯度累积到对应张量的 ``.grad`` 属性中
# - 使用链式法则，一直传播到叶子张量
#
# .. note::
#   **PyTorch 中的 DAG 是动态的**
#   需要注意的一点是，这个图每次都会从头重新创建；
#   每次调用 ``.backward()`` 之后，autograd 都会开始填充一个新的图。
#   这正是你能够在模型中使用控制流语句的原因；如果有需要，
#   你可以在每次迭代时改变张量的形状、大小和操作。

######################################################################
# 选读：张量梯度与雅可比乘积
# --------------------------
#
# 在很多情况下，我们会有一个标量损失函数，并且需要计算它相对于
# 某些参数的梯度。不过，也存在输出函数是任意张量的情况。在这种情况下，
# PyTorch 允许你计算所谓的**雅可比乘积**，而不是真正完整的梯度。
#
# 对于向量函数 :math:`\vec{y}=f(\vec{x})`，其中
# :math:`\vec{x}=\langle x_1,\dots,x_n\rangle` 且
# :math:`\vec{y}=\langle y_1,\dots,y_m\rangle`，则
# :math:`\vec{y}` 相对于 :math:`\vec{x}` 的梯度可以由
# **雅可比矩阵**表示：
#
# .. math::
#
#
#    J=\left(\begin{array}{ccc}
#       \frac{\partial y_{1}}{\partial x_{1}} & \cdots & \frac{\partial y_{1}}{\partial x_{n}}\\
#       \vdots & \ddots & \vdots\\
#       \frac{\partial y_{m}}{\partial x_{1}} & \cdots & \frac{\partial y_{m}}{\partial x_{n}}
#       \end{array}\right)
#
# PyTorch 并不会直接去计算整个雅可比矩阵本身，而是允许你对给定的
# 输入向量 :math:`v=(v_1 \dots v_m)` 计算**雅可比乘积**
# :math:`v^T\cdot J`。这是通过将 :math:`v` 作为参数传给
# ``backward`` 来实现的。:math:`v` 的大小应当与我们希望对其
# 进行该乘积计算的原始张量大小一致：
#

inp = torch.eye(4, 5, requires_grad=True)
out = (inp + 1).pow(2).t()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"First call\n{inp.grad}")
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nSecond call\n{inp.grad}")
inp.grad.zero_()
out.backward(torch.ones_like(out), retain_graph=True)
print(f"\nCall after zeroing gradients\n{inp.grad}")


######################################################################
# 注意，当我们第二次使用相同参数调用 ``backward`` 时，
# 梯度的值会发生变化。这是因为在执行 ``backward`` 传播时，
# PyTorch 会**累积梯度**，也就是说，计算得到的梯度值会被加到
# 计算图所有叶子节点的 ``grad`` 属性上。如果你想计算正确的梯度，
# 就需要事先将 ``grad`` 属性清零。在实际训练中，*优化器* 会帮助
# 我们完成这件事。

######################################################################
# .. note:: 之前我们调用 ``backward()`` 时没有传入参数。
#           这本质上等价于调用 ``backward(torch.tensor(1.0))``，
#           对于标量值函数（例如神经网络训练中的损失）来说，
#           这是一种很有用的梯度计算方式。
#

######################################################################
# --------------
#

#################################################################
# 延伸阅读
# ~~~~~~~~
# - `Autograd 机制 <https://pytorch.org/docs/stable/notes/autograd.html>`_
