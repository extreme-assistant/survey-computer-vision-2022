论文打包下载：http://m6z.cn/5AWNia



# 目录
- [**分类|识别相关(15篇)**](#分类识别相关)
- [**检测相关**](#检测相关)
- [**Transformer**](#transformer)
- [**医学相关**](#医学相关)
- [**多模态**](#多模态)
- [跟踪](#跟踪)
- [**超分辨率|去噪|去模糊|去雾**](#超分辨率去噪去模糊去雾)
- [人脸](#人脸)
- [**视觉解释|视频理解VQA|caption等**](#视觉解释视频理解vqacaption等)
- [**时序|行为识别|姿态|视频|运动估计**](#时序行为识别姿态视频运动估计)
- [**自动驾驶|车辆|车道检测**](#自动驾驶车辆车道检测)
- [图像分割](#图像分割)
- [gan](#gan)
- [**NAS模型搜索**](#nas模型搜索)
- [表征学习](#表征学习)
- [**半弱无监督|主动学习|不确定性**](#半弱无监督主动学习不确定性)
- [**Zero/Few Shot|迁移|域适配|自适应**](#zerofew-shot迁移域适配自适应)
- [**点云|SLAM|雷达|激光|深度RGBD相关**](#点云slam雷达激光深度rgbd相关)
- [**3D|3D重建**](#3d3d重建)
- [Attention注意力](#attention注意力)
- [**裁剪|量化|加速|压缩相关**](#裁剪量化加速压缩相关)
- [语义分割](#语义分割)
- [**其他神经网络|深度学习|模型|建模**](#其他神经网络深度学习模型建模)
- [**蒸馏|知识提取**](#蒸馏知识提取)
- [**其他**](#其他)

<br><br>

<a name="分类识别相关"/>


## **分类|识别相关**

【1】 A Survey on Long-Tailed Visual Recognition<br>
**标题**：长尾视觉识别研究综述<br>
**链接**：https://arxiv.org/abs/2205.13775<br>

**作者**：Lu Yang,He Jiang,Qing Song,Jun Guo<br>
**机构**： Beijing University of Posts and Telecommunications(priv<br>
**备注**：Accepted for publication in International Journal of Computer Vision (IJCV)<br>
**摘要**：对数据的严重依赖是目前限制深度学习发展的主要原因之一。数据质量直接决定了深度学习模型的效果，而长尾分布是影响数据质量的因素之一。由于自然界中普遍存在幂律，长尾现象很普遍。在这种情况下，深度学习模型的表现往往由头班主导，而尾班的学习严重不足。为了让所有班级都能充分学习，许多研究人员已经研究并初步解决了长尾问题。本文针对长尾数据分布带来的问题，对具有代表性的长尾视觉识别数据集进行了梳理，总结了一些主流的长尾研究。具体而言，我们从表征学习的角度将这些研究归纳为十大类，并概述了每一类的亮点和局限性。此外，我们还研究了评估不平衡的四个量化指标，并建议使用基尼系数来评估数据集的长尾性。基于基尼系数，我们对过去十年提出的20个广泛使用的大规模视觉数据集进行了定量研究，发现长尾现象普遍存在，尚未得到充分研究。最后，我们为长尾学习的发展提供了几个未来的方向，以便为读者提供更多的想法。

<br><br>

【2】 Deep transfer learning for image classification: a survey<br>
**标题**：深度迁移学习在图像分类中的研究进展<br>
**链接**：https://arxiv.org/abs/2205.09904<br>

**作者**：Jo Plested,Tom Gedeon<br>
**机构**：School of Engineering and Information Technology, University of New South Wales, Northcott Drive, Campbell, ACT, Australia., Optus Centre for Artificial Intelligence, Curtin University, Kent Street, Bentley, WA, Australia.<br>
**摘要**：近年来，卷积神经网络（CNN）和Transformer等深层神经网络在图像分类方面取得了许多成功。一直以来，图像分类的最佳实践是在大量标记数据上训练大型深层模型。然而，在许多实际场景中，无法满足获取最佳性能所需的大量训练数据。在这些情况下，迁移学习可以帮助提高绩效。到目前为止，还没有全面审查深度迁移学习的调查，因为它与图像分类总体相关。然而，最近发布了一些关于深度迁移学习的一般调查，以及与特定专业目标图像分类任务相关的调查。我们认为，整理所有现有知识，分析和讨论总体模式，对于该领域未来的进展至关重要。在这项调查中，我们正式定义了深度迁移学习及其试图解决的与图像分类相关的问题。我们调查了该领域的现状，并确定了最近取得的进展。我们将展示当前知识的差距，并就如何在该领域取得进展以填补这些知识差距提出建议。我们提出了一种新的分类方法，将转移学习应用于图像分类。这种分类法使我们更容易看到迁移学习在哪些方面有效，在哪些方面未能发挥其潜力的总体模式。这也使我们能够提出问题所在以及如何更有效地利用这些问题。我们表明，在这种新的分类法下，当考虑到源和目标数据集以及使用的技术时，许多迁移学习被证明无效甚至阻碍性能的应用程序是可以预期的。

<br><br>

【3】 A Close Look into Human Activity Recognition Models using Deep Learning<br>
**标题**：基于深度学习的人类活动识别模型研究<br>
**链接**：https://arxiv.org/abs/2204.13589<br>

**作者**：Wei Zhong Tee,Rushit Dave,Naeem Seliya,Mounika Vanamala<br>
**机构**：Undergraduate Computer Science Major, University of Wisconsin-Eau Claire, Eau Claire, United States, Jim Seliya, Assistant Professor, Computer Science Dept<br>
**摘要**：使用深度学习技术的人类活动识别越来越受欢迎，因为它在识别复杂任务时具有很高的效率，并且与传统的机器学习技术相比成本相对较低。本文综述了一些基于深度学习体系结构的最新人类活动识别模型，这些模型的层次包括卷积神经网络（CNN）、长短时记忆（LSTM）或混合系统的多种类型。分析概述了如何实施这些模型以最大限度地提高其有效性，以及它面临的一些潜在限制。

<br><br>

【4】 Periocular Biometrics and its Relevance to Partially Masked Faces: A Survey<br>
**标题**：眼周生物识别技术及其与部分蒙面人脸相关性的研究进展<br>
**链接**：https://arxiv.org/abs/2203.15203<br>

<br><br>

【5】 Semi-supervised Deep Learning for Image Classification with Distribution Mismatch: A Survey<br>
**标题**：半监督深度学习在分布失配图像分类中的研究进展<br>
**链接**：https://arxiv.org/abs/2203.00190<br>

**作者**：Saul Calderon-Ramirez,Shengxiang Yang,David Elizondo<br>
**备注**：Submission to IEEE Transactions on AI<br>
**摘要**：深度学习方法已被应用于多个不同的领域，在图像识别应用中取得了显著的成功，如材料质量控制、医学成像、自动驾驶等。深度学习模型依赖于大量的标记观察来训练前瞻性模型。这些模型由数百万个要估计的参数组成，增加了对更多训练观测的需求。通常，收集标记的数据观测值很昂贵，这使得深度学习模型的使用并不理想，因为该模型可能会过度拟合数据。在半监督环境中，使用未标记数据来提高带有小标记数据集的模型的准确性和泛化水平。然而，在许多情况下，可能会有不同的未标记数据源可用。这增加了标记数据集和未标记数据集之间存在显著分布不匹配的风险。这种现象可能会对典型的半监督深度学习框架造成相当大的性能影响，这些框架通常假设标记和未标记的数据集都来自类似的分布。因此，本文研究了用于图像识别的半监督深度学习的最新方法。重点介绍了半监督深度学习模型，该模型旨在处理标记和未标记数据集之间的分布不匹配。我们应对开放性挑战，旨在鼓励社区应对这些挑战，并克服现实世界使用环境下传统深度学习管道的高数据需求。

<br><br>

【6】 Continuous Human Action Recognition for Human-Machine Interaction: A Review<br>
**标题**：人机交互中的连续人类行为识别研究进展<br>
**链接**：https://arxiv.org/abs/2202.13096<br>

**作者**：Harshala Gammulle,David Ahmedt-Aristizabal,Simon Denman,Lachlan Tychsen-Smith,Lars Petersson,Clinton Fookes<br>
**备注**：Preprint submitted to ACM Computing Surveys<br>
**摘要**：随着数据驱动的机器学习研究的发展，人们提出了各种各样的预测模型来捕获时空特征，以便分析视频流。对于需要实时人机交互的应用程序来说，识别输入视频中的动作和检测动作转换是一项具有挑战性但必不可少的任务。通过回顾大量近期相关文献，我们深入分析、解释和比较了动作分割方法，并详细介绍了大多数最先进方法中使用的特征提取和学习策略。我们讨论了目标检测和跟踪技术的性能对人体动作分割方法的影响。我们研究了此类模型在现实场景中的应用，并讨论了一些局限性和关键研究方向，以提高可解释性、通用性、优化和部署。

<br><br>

【7】 Person Re-identification: A Retrospective on Domain Specific Open Challenges and Future Trends<br>
**标题**：人的再认同：特定领域开放挑战与未来趋势回顾<br>
**链接**：https://arxiv.org/abs/2202.13121<br>

**作者**：Asmat Zahra,Nazia Perwaiz,Muhammad Shahzad,Muhammad Moazam Fraz<br>
**摘要**：人员重新识别（re ID）是自动视觉监控系统的主要组成部分之一。它旨在自动识别/搜索具有非重叠视野的多摄像头网络中的人员。由于其潜在的各种应用和研究意义，近年来提出了大量基于深度学习的REID方法。然而，存在着一些与视觉相关的挑战，例如遮挡、姿势尺度和视点变化、背景杂波、人员错位和跨摄像机模式的跨域泛化，这使得重新识别问题仍然远未得到解决。大多数提议的方法直接或间接旨在解决一个或多个现有挑战。在这种情况下，需要对当前解决这些挑战的REID方法进行全面审查，以分析和关注特定方面，以进一步推进。目前还不存在这样一个有针对性的综述，因此在本文中，我们对2015-21年间的230多篇论文进行了系统的挑战性文献调查。首次提出了此类调查，从这种面向解决方案的角度对人员识别方法进行了审查。此外，我们还介绍了各自研究领域中几种不同的突出发展趋势，这将为正在进行的个人识别研究提供一个有远见的视角，并最终帮助开发实际的解决方案。

<br><br>

【8】 Gait Recognition Based on Deep Learning: A Survey<br>
**标题**：基于深度学习的步态识别研究综述<br>
**链接**：https://arxiv.org/abs/2201.03323<br>

**作者**：Claudio Filipi Gonçalves dos Santos,Diego de Souza Oliveira,Leandro A. Passos,Rafael Gonçalves Pires,Daniel Felipe Silva Santos,Lucas Pascotti Valem,Thierry P. Moreira,Marcos Cleison S. Santana,Mateus Roder,João Paulo Papa,Danilo Colombo<br>
**机构**：and Eldorado Research Institute, Brazil, State University - UNESP, Brazil, In general, operate appropriately. Instead, such systems should be aware of malicious procedures for unauthorized access<br>
**摘要**：一般来说，基于生物测量学的控制系统可能不依赖于个体预期行为或合作来适当操作。相反，此类系统应了解未经授权访问尝试的恶意程序。文献中的一些工作建议通过步态识别方法来解决这个问题。这些方法的目的是通过内在的可感知特征来识别人类，不管穿着什么衣服或配饰。尽管这一问题意味着一个相对较长的挑战，但为处理该问题而开发的大多数技术都存在与特征提取和低分类率等问题相关的一些缺点。然而，基于深度学习的方法最近作为一套强大的工具出现，用于处理几乎任何与图像和计算机视觉相关的问题，也为步态识别提供了最重要的结果。因此，这项工作提供了一份关于通过步态识别进行生物特征检测的最新工作的调查汇编，重点是深入学习方法，强调其优点，并揭示其缺点。此外，它还提供了用于处理相关约束的数据集、方法和体系结构的分类和特征描述。

<br><br>

【9】 A Survey on Face Recognition Systems<br>
**标题**：人脸识别系统综述<br>
**链接**：https://arxiv.org/abs/2201.02991<br>

**作者**：Jash Dalvi,Sanket Bafna,Devansh Bagaria,Shyamal Virnodkar<br>
**机构**：KJSIEIT, Mumbai, India, Asst. Professor, Computer Engineering Department, K. J. Somaiya Institute of Engineering and Information Technology<br>
**摘要**：人脸识别已被证明是最成功的技术之一，并已影响到异构领域。由于其基于卷积的体系结构，深度学习已被证明是计算机视觉任务中最成功的。自从深度学习出现以来，人脸识别技术的准确率有了很大的提高。本文综述了一些最有影响力的人脸识别系统。本文首先概述了一个通用的人脸识别系统。其次，调查涵盖了各种网络架构和训练损失，这些都产生了重大影响。最后，本文讨论了用于评估人脸识别系统性能的各种数据库。

<br><br>

【10】 Beyond the Visible: A Survey on Cross-spectral Face Recognition<br>
**标题**：看不见的之外：跨光谱人脸识别研究综述<br>
**链接**：https://arxiv.org/abs/2201.04435<br>

**作者**：David Anghelone,Cunjian Chen,Arun Ross,Antitza Dantcheva<br>
**机构**： Ross are with the Department of Computer Scienceand Engineering at Michigan State University (MSU)<br>
**摘要**：交叉光谱人脸识别（CFR）旨在识别个人，其中比较的人脸图像来自不同的传感模式，例如红外和可见光。虽然CFR固有地比经典人脸识别更具挑战性，因为与模态间隙相关的面部外观存在显著差异，但它在照明有限或具有挑战性的场景中以及在呈现攻击存在的情况下更具优势。与卷积神经网络（CNN）相关的人工智能的最新进展使CFR的性能显著提高。基于此，本次调查的贡献有三个方面。我们首先对CFR进行形式化描述，然后介绍具体的相关应用，从而对CFR进行概述，旨在比较在不同光谱中捕获的人脸图像。其次，我们探索适合识别的光谱波段，并讨论最新的CFR方法，重点放在深层神经网络上。特别是，我们回顾了用于提取和比较异构特征以及数据集的技术。我们列举了不同光谱和相关算法的优点和局限性。最后，我们讨论了研究挑战和未来的研究方向。

<br><br>

【11】 A Survey of Historical Document Image Datasets<br>
**标题**：历史文献图像数据集综述<br>
**链接**：https://arxiv.org/abs/2203.08504<br>

**作者**：Konstantina Nikolaidou,Mathias Seuret,Hamam Mokayed,Marcus Liwicki<br>
**机构**：EISLAB Machine Learning Group, Luleå University of Technology, Aurorum , Luleå, Norrbotten, Sweden., Pattern Recognition Lab Computer Vision Group, Friedrich-Alexander-Universität, Martensstr. , Erlangen, Bavaria, Germany.<br>
**摘要**：本文对用于文档图像分析的图像数据集进行了系统的文献综述，重点介绍了历史文档，如手写手稿和早期印刷品。为历史文档分析找到合适的数据集是促进使用不同机器学习算法进行研究的关键先决条件。然而，由于实际数据种类繁多（例如脚本、任务、日期、支持系统和退化量），数据和标签表示的格式不同，评估过程和基准也不同，因此找到合适的数据集是一项困难的任务。这项工作填补了这一空白，对现有数据集进行了元研究。经过一个系统的选择过程（根据PRISMA指南），我们选择了56项研究，这些研究是基于不同的因素选择的，比如发表年份、文章中采用的方法数量、所选算法的可靠性、数据集大小和期刊发行量。我们通过将每个研究分配到三个预定义任务中的一个来进行总结：文档分类、布局结构或语义分析。我们展示了每个数据集的统计数据、文档类型、语言、任务、输入视觉方面和地面真相信息。此外，我们还提供了这些论文或近期竞赛的基准任务和结果。我们将进一步讨论这一领域的差距和挑战。我们提倡提供通用格式的转换工具（例如，计算机视觉任务的COCO格式），并始终提供一组评估指标，而不是一个，以使研究结果具有可比性。

<br><br>

【12】 A Survey of Robust Adversarial Training in Pattern Recognition: Fundamental, Theory, and Methodologies<br>
**标题**：模式识别中稳健的对抗性训练：基础、理论和方法综述<br>
**链接**：https://arxiv.org/abs/2203.14046<br>

**作者**：Zhuang Qian,Kaizhu Huang,Qiu-Feng Wang,Xu-Yao Zhang<br>
**机构**：School of Advanced Technology of Xi’an Jiaotong-Liverpool University, Institute of Applied Physical Sciences and Engineering, Duke Kunshan University, Institute of Automation, Chinese Academy of Sciences; School of Artificial Intelligence<br>
**摘要**：在过去的几十年里，深度神经网络在机器学习、计算机视觉和模式识别方面取得了显著的成功。然而，最近的研究表明，神经网络（浅层和深层）可能很容易被某些被称为对抗性示例的不易察觉的扰动输入样本所愚弄。近年来，由于神经网络的广泛应用，可能会引入现实世界中的威胁，因此此类安全漏洞引发了大量研究。为了解决对抗性示例的鲁棒性问题，尤其是在模式识别中，鲁棒对抗性训练已成为主流之一。各种想法、方法和应用在该领域蓬勃发展。然而，对对抗训练的深入理解，包括特征、解释、理论以及不同模式之间的联系，仍然很难实现。在本文中，我们提出了一个全面的调查，试图提供一个系统和结构化的调查稳健对抗训练模式识别。我们从基础知识开始，包括对抗性示例的定义、符号和属性。然后，我们介绍了一个针对对抗性样本进行防御的统一理论框架——稳健的对抗性训练，并对为什么对抗性训练可以导致模型稳健进行了可视化和解释。对抗性训练和其他传统学习理论之间也将建立联系。之后，我们以结构化的方式总结、回顾和讨论各种对抗性攻击和防御/训练算法的方法。最后，我们对对抗训练进行了分析、展望和评论。

<br><br>

【13】 A Comprehensive Survey on Deep Gait Recognition: Algorithms, Datasets and Challenges<br>
**标题**：深度步态识别研究综述：算法、数据集和挑战<br>
**链接**：https://arxiv.org/abs/2206.13732<br>

**作者**：Chuanfu Shen,Shiqi Yu,Jilong Wang,George Q. Huang,Liang Wang<br>
**机构**： and alsowith Department of Industrial and Manufacturing Systems Engineering, TheUniversity of Hong Kong, Shiqi Yu is with the Department of Computer Science and Engineering, Southern University of Science and Technology<br>
**摘要**：步态识别旨在通过视觉摄像头识别远处的人。随着深度学习的出现，通过利用深度学习技术，步态识别在许多场景中取得了令人鼓舞的成功。然而，对视频监控的日益增长的需求带来了更多的挑战，包括各种变化下的鲁棒识别、步态序列中的运动信息建模、协议变化导致的不公平性能比较、生物特征安全和隐私保护。本文对步态识别的深度学习进行了全面的综述。我们首先介绍了步态识别从传统算法到深度模型的发展历程，为步态识别系统的整个工作流程提供了明确的知识。然后从深度表征和结构的角度讨论了步态识别的深度学习，并对其进行了深入的总结。具体来说，深度步态表示分为静态和动态特征，而深度架构包括单流和多流架构。根据我们提出的新颖分类法，它有助于提供灵感和促进对深度步态识别的感知。此外，我们还对所有基于视觉的步态数据集和性能分析进行了全面总结。最后，本文讨论了一些具有重大潜在前景的开放性问题。

<br><br>

【14】 Going Deeper than Tracking: a Survey of Computer-Vision Based Recognition of Animal Pain and Affective States<br>
**标题**：比跟踪更深入：基于计算机视觉的动物疼痛和情感状态识别综述<br>
**链接**：https://arxiv.org/abs/2206.08405<br>

**作者**：Sofia Broomé,Marcelo Feighelstein,Anna Zamansky,Gabriel Carreira Lencioni,Pia Haubro Andersen,Francisca Pessanha,Marwa Mahmoud,Hedvig Kjellström,Albert Ali Salah<br>
**摘要**：动物运动跟踪和姿势识别的进展已经成为动物行为研究中的一个游戏规则改变者。最近，越来越多的工作比追踪“更深入”，并致力于自动识别动物的内部状态，如情绪和疼痛，目的是改善动物福利，使这成为该领域系统化的及时时机。本文对基于计算机视觉的动物情感状态和疼痛识别研究进行了全面综述，涉及面部和身体行为分析。我们总结了到目前为止在本主题中所做的工作——从不同维度对其进行分类，突出挑战和研究差距，并提供推进该领域的最佳实践建议，以及一些未来的研究方向。

<br><br>

【15】 Recent Advances in Scene Image Representation and Classification<br>
**标题**：场景图像表示与分类的研究进展<br>
**链接**：https://arxiv.org/abs/2206.07326<br>

**作者**：Chiranjibi Sitaula,Tej Bahadur Shahi,Faezeh Marzbanrad<br>
**机构**：Department of Electrical and Computer Systems Engineering, Wellington, Rd, Clayton, VIC, Australia, School of Engineering and Technology, CQUniversity, Yaamba, Rd, Rockhampton, QLD, Australia, Central Department of Computer Science and IT, Tribhuvan<br>
**备注**：This paper is under review in Computer Science Review (Elsevier) journal. This article may be deleted or updated based on the polices of the journal<br>
**摘要**：随着深度学习算法的兴起，大数据（如SUN-397）上的场景图像表示方法在分类方面取得了显著的性能提升。然而，由于场景图像大多是复杂的，具有较高的类内相似度和类间相似度问题，因此其性能仍然有限。为了解决这些问题，文献中提出了几种方法，各有其优点和局限性。有必要对以前的工作进行详细研究，以了解它们在图像表示和分类方面的优缺点。在本文中，我们回顾了目前广泛用于图像分类的场景图像表示方法。为此，我们首先利用迄今为止文献中提出的开创性现有方法设计分类学。接下来，我们从定性（例如，输出质量、优缺点等）和定量（例如，准确性）两方面比较它们的性能。最后，我们推测了场景图像表示任务的主要研究方向。总的来说，本调查为基于传统计算机视觉（CV）的方法、基于深度学习（DL）的方法和基于搜索引擎（SE）的方法提供了最新场景图像表示方法的深入见解和应用。

<br><br>

【16】 Two Decades of Bengali Handwritten Digit Recognition: A Survey<br>
**标题**：孟加拉手写数字识别二十年的调查<br>
**链接**：https://arxiv.org/abs/2206.02234<br>

**作者**：A. B. M. Ashikur Rahman,Md. Bakhtiar Hasan,Sabbir Ahmed,Tasnim Ahmed,Md. Hamjajul Ashmafee,Mohammad Ridwan Kabir,Md. Hasanul Kabir<br>
**机构**：(Member, IEEE), Department of Computer Science and Engineering, Islamic University of Technology, Dhaka, Bangladesh<br>
**备注**：This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible. 35 pages, 20 figures, 12 tables<br>
**摘要**：手写数字识别（HDR）是光学字符识别（OCR）领域中最具挑战性的任务之一。无论语言如何，HDR都存在一些固有的挑战，这些挑战主要是由于个人、书写媒介和环境的书写风格不同，无法在重复书写任何数字时保持相同的笔画等。此外，特定语言数字的结构复杂性可能导致HDR的模糊场景。多年来，研究人员开发了许多脱机和联机HDR管道，其中不同的图像处理技术与传统的基于机器学习（ML）和/或基于深度学习（DL）的体系结构相结合。尽管英语、阿拉伯语、印第安语、波斯语、汉语等语言的文献中存在关于HDR的广泛回顾研究证据，但很少有关于孟加拉HDR（BHDR）的调查，缺乏对挑战、潜在识别过程和未来可能方向的全面分析。本文分析了孟加拉手写数字的特点和固有的歧义性，以及二十年来最先进的数据集和离线BHDR方法的综合见解。此外，还详细讨论了几个涉及BHDR的实际应用特定研究。本文还将为对离线BHDR背后的科学感兴趣的研究人员提供一份概要，鼓励探索新的相关研究途径，进一步提高不同应用领域中孟加拉手写数字的离线识别能力。

<br><br>

【17】 A Survey on Video Action Recognition in Sports: Datasets, Methods and Applications<br>
**标题**：运动视频动作识别研究综述：数据集、方法及应用<br>
**链接**：https://arxiv.org/abs/2206.01038<br>

**作者**：Fei Wu,Qingzhong Wang,Jian Bian,Haoyi Xiong,Ning Ding,Feixiang Lu,Jun Cheng,Dejing Dou<br>
**机构**：Fei Wu and Ning Ding are with the Department of Physical Education, Peking University<br>
**备注**：26 pages. The toolbox is available at this https URL<br>
**摘要**：为了理解人类行为，基于视频的动作识别是一种常用的方法。与基于图像的动作识别相比，视频提供了更多的信息。为了减少动作的模糊性，在过去的十年中，许多研究集中在数据集、新模型和学习方法上，将视频动作识别提高到了一个更高的水平。然而，仍然存在挑战和未解决的问题，尤其是在体育分析领域，数据收集和标记更加复杂，需要体育专业人员对数据进行注释。此外，这些动作可能非常快，很难识别。此外，在足球和篮球这样的团队运动中，一个动作可能涉及多个球员，为了正确识别他们，我们需要分析所有球员，这是相对复杂的。在这篇文章中，我们提出了一个调查视频动作识别体育分析。我们推出十多种运动，包括足球、篮球、排球、曲棍球等团队运动和花样滑冰、体操、乒乓球、网球、跳水和羽毛球等个人运动。然后，我们将现有的众多运动分析框架与团队运动和个人运动中视频动作识别的现状进行了比较。最后，我们讨论了这一领域的挑战和未解决的问题，为了便于体育分析，我们开发了一个使用飞浆的工具箱，该工具箱支持足球、篮球、乒乓球和花样滑冰动作识别。

<br><br>

【18】 An Overview of Privacy-enhancing Technologies in Biometric Recognition<br>
**标题**：生物特征识别中的隐私增强技术综述<br>
**链接**：https://arxiv.org/abs/2206.10465<br>

**作者**：Pietro Melzi,Christian Rathgeb,Ruben Tolosana,Ruben Vera-Rodriguez,Christoph Busch<br>
**机构**： Universidad Autonoma de Madrid<br>
**摘要**：隐私增强技术是实现基本数据保护原则的技术。关于生物特征识别，引入了不同类型的隐私增强技术，以保护存储的生物特征数据，这些数据通常被归类为敏感数据。在这方面，提出了各种分类法和概念分类，并开展了标准化活动。然而，这些努力主要集中在隐私增强技术的某些子类别上，因此缺乏普遍性。这项工作概述了在统一框架中用于生物特征识别的隐私增强技术的概念。在每个处理步骤中，都会详细强调现有概念之间的关键方面和差异。讨论了现有方法的基本特性和局限性，并与数据保护技术和原则相关。此外，还介绍了评估生物识别隐私增强技术的场景和方法。本文旨在作为生物特征数据保护领域的切入点，面向经验丰富的研究人员以及非专家。

<br><br>

<a name="检测相关"/>

# **检测相关**

【1】 Detecting and Understanding Harmful Memes: A Survey<br>
**标题**：有害模因的发现和理解：一项调查<br>
**链接**：https://arxiv.org/abs/2205.04274<br>

**作者**：Shivam Sharma,Firoj Alam,Md. Shad Akhtar,Dimitar Dimitrov,Giovanni Da San Martino,Hamed Firooz,Alon Halevy,Fabrizio Silvestri,Preslav Nakov,Tanmoy Chakraborty<br>
**机构**：Qatar Computing Research Institute, HBKU, Qatar,IIIT-Delhi, India,Sapienza University of Rome, Italy,Sofia University, Bulgaria,University of Padova, Italy,Facebook AI, USA,Wipro AI Labs<br>
**备注**：Accepted at IJCAI-ECAI 2022 (Survey Track)<br>
**摘要**：在线有害内容的自动识别是社交媒体平台、决策者和社会关注的主要问题。研究人员研究了文本、视觉和音频内容，但通常是孤立的。然而，有害内容往往会结合多种形式，比如模因，由于其病毒性，它们特别令人感兴趣。考虑到这一点，我们在这里提供了一个全面的调查，重点是有害的模因。基于对近期文献的系统分析，我们首先提出了一种新的有害模因类型，然后重点介绍和总结了相关的研究现状。一个有趣的发现是，许多类型的有害模因没有得到真正的研究，例如，以自我伤害和极端主义为特征的模因，部分原因是缺乏合适的数据集。我们进一步发现，现有的数据集大多捕捉了多类场景，这些场景不包括模因所能代表的情感谱。另一个观察结果是，模因可以通过不同语言的重新包装在全球传播，它们也可以是多语言的，融合不同的文化。最后，我们强调了与多模态符号学、技术约束和非琐碎的社会参与相关的几个挑战，并提出了几个开放性方面，如描述在线危害，实证研究相关框架和辅助干预措施，我们相信这些将激励和推动未来的研究

<br><br>

【2】 Computer Vision for Road Imaging and Pothole Detection: A State-of-the-Art Review of Systems and Algorithms<br>
**标题**：用于道路成像和坑洞检测的计算机视觉：系统和算法的最新进展<br>
**链接**：https://arxiv.org/abs/2204.13590<br>

**作者**：Nachuan Ma,Jiahe Fan,Wenshuo Wang,Jin Wu,Yu Jiang,Lihua Xie,Rui Fan<br>
**机构**：∗, Department of Control Science and Engineering, Tongji University, Shanghai , P. R. China,Department of Civil Engineering, McGill University, Montr´eal, QC H,A ,C, Canada,Department of Electronics and Computer Engineering, the Hong Kong University of<br>
**备注**：accepted to Transportation Safety and Environment<br>
**摘要**：20多年来，计算机视觉算法广泛应用于三维道路成像和坑洞检测。尽管如此，对于最先进的计算机视觉技术，尤其是为解决这些问题而开发的深度学习模型，还缺乏系统的调查文章。本文首先介绍了用于二维和三维道路数据采集的传感系统，包括摄像头、激光扫描仪和Microsoft Kinect。随后，全面全面地回顾了SoTA计算机视觉算法，包括（1）经典的二维图像处理，（2）三维点云建模和分割，以及（3）为道路坑洞检测开发的机器/深度学习。本文还讨论了基于计算机视觉的道路坑洞检测方法存在的挑战和未来的发展趋势：经典的基于二维图像处理和基于三维点云建模和分割的方法已经成为历史；卷积神经网络（CNN）已经证明了令人信服的道路坑洞检测结果，并有望随着未来在多模态语义分割的自/无监督学习方面的进步打破瓶颈。我们相信，这项调查可以作为开发下一代道路状况评估系统的实际指导。

<br><br>

【3】 A Survey on Unsupervised Industrial Anomaly Detection Algorithms<br>
**标题**：无监督工业异常检测算法综述<br>
**链接**：https://arxiv.org/abs/2204.11161<br>

**作者**：Yajie Cui,Zhaoxiang Liu,Shiguo Lian<br>
**机构**：AI Innovation and Application Center, China Unicom, Beijing, China.<br>
**摘要**：异常缺陷检测已成为工业生产过程中不可缺少的一部分。在以往的研究中，传统的异常检测算法大多属于有监督学习的范畴，而在大多数实际应用场景中，无监督情况更为常见。因此，在过去的几年里，逐渐无监督的异常检测一直是许多研究的主题。在本综述中，我们全面介绍了新提出的视觉异常检测方法。我们希望它能帮助研究界以及行业领域建立更广泛和跨领域的视角。

<br><br>

【4】 A Survey of Robust 3D Object Detection Methods in Point Clouds<br>
**标题**：点云环境下稳健的三维目标检测方法综述<br>
**链接**：https://arxiv.org/abs/2204.00106<br>

**作者**：Walter Zimmer,Emec Ercelik,Xingcheng Zhou,Xavier Jair Diaz Ortiz,Alois Knoll<br>
**摘要**：这项工作的目的是回顾最先进的基于激光雷达的三维物体检测方法、数据集和挑战。我们描述了新的数据增强方法、采样策略、激活函数、注意机制和正则化方法。此外，我们还列出了最近引入的规范化方法、学习速率计划和损失函数。此外，我们还介绍了10种新型自动驾驶数据集的优点和局限性。我们在KITTI、nuScenes和Waymo数据集上评估了新型3D对象检测器，并展示了它们的准确性、速度和鲁棒性。最后，我们提到了激光雷达点云中三维目标检测的当前挑战，并列出了一些尚未解决的问题。

<br><br>

【5】 An Empirical Study and Comparison of Recent Few-Shot Object Detection Algorithms<br>
**标题**：当前几种Few-Shot目标检测算法的实验研究与比较<br>
**链接**：https://arxiv.org/abs/2203.14205<br>

**作者**：Tianying Liu,Lu Zhang,Yang Wang,Jihong Guan,Yanwei Fu,Shuigeng Zhou<br>
**机构**：Member, IEEE<br>
**摘要**：通用目标检测（GOD）任务已由最近的深层神经网络成功解决，该网络由来自一些常见类的大量带注释的训练样本进行训练。然而，将这些对象检测器推广到新的长尾对象类仍然是非常重要的，因为它只有很少的标记训练样本。为此，少数镜头物体检测（FSOD）最近成为热门话题，因为它模仿了人类的学习能力，并智能地将学习到的一般物体知识从常见的重尾类转移到新的长尾类物体。尤其是近年来，随着各种基准、支柱和方法的提出，这一新兴领域的研究蓬勃发展。为了回顾这些FSOD工作，有几篇有见地的FSOD调查文章对它们进行了系统的研究，并将它们作为微调/迁移学习和元学习方法进行了比较。相比之下，我们从新的角度和分类法比较了这些FSOD算法，即面向数据、面向模型和面向算法的算法。因此，对FSOD的最新成果进行了实证研究和比较。此外，我们还分析了这些方法的技术挑战、优缺点，并展望了FSOD的未来发展方向。具体来说，我们对FSOD进行了概述，包括问题定义、常见数据集和评估协议。基于先验知识在新类目标检测中的作用，提出了一种新的分类方法。根据这一分类法，我们对FSOD的进展进行了系统的回顾。最后，对性能、挑战和未来方向进行了进一步讨论。

<br><br>

【6】 Detection, Recognition, and Tracking: A Survey<br>
**标题**：检测、识别和跟踪：综述<br>
**链接**：https://arxiv.org/abs/2203.11900<br>

**作者**：Shiyao Chen,Dale Chen-Song<br>
**摘要**：对人类来说，目标检测、识别和跟踪是与生俱来的。它们为人类提供了感知环境和环境中物体的能力。然而，这种能力在计算机中并不适用。在计算机视觉和多媒体领域，检测、识别和跟踪图像和/或视频中的物体变得越来越重要。其中许多应用程序，如人脸识别、监控、动画，都用于跟踪特征和/或人物。然而，这些任务对于计算机来说很难有效地完成，因为有大量数据需要解析。因此，人们需要并研究许多技术和算法来实现类似人类的感知。在这篇文献综述中，我们重点介绍了一些新的目标检测和识别技术，以及如何将跟踪算法应用于检测到的特征来跟踪目标的运动。

<br><br>

【7】 A Survey of Surface Defect Detection of Industrial Products Based on A Small Number of Labeled Data<br>
**标题**：基于少量标注数据的工业产品表面缺陷检测综述<br>
**链接**：https://arxiv.org/abs/2203.05733<br>

**作者**：Qifan Jin,Li Chen<br>
**机构**： ( 2) Institute of Physical Education (Main Campus), Zhengzhou University) Abstract<br>
**摘要**：基于视觉感知的表面缺陷检测方法在工业质量检测中得到了广泛的应用。由于缺陷数据不易获取，对大量缺陷数据进行标注会浪费大量人力物力。因此，本文综述了基于少量标记数据的工业产品表面缺陷检测方法，该方法分为基于传统图像处理的工业产品表面缺陷检测方法和基于深度学习的适合于少量标注数据的工业产品表面缺陷检测方法。传统的基于图像处理的工业产品表面缺陷检测方法分为统计方法、光谱方法和模型方法。基于深度学习的工业产品表面缺陷检测方法适用于少量标注数据，分为基于数据增强、基于转移学习、基于模型的微调、半监督、弱监督和无监督。

<br><br>

【8】 A Survey on Masked Facial Detection Methods and Datasets for Fighting Against COVID-19<br>
**标题**：对抗冠状病毒的蒙面人脸检测方法和数据集的研究进展<br>
**链接**：https://arxiv.org/abs/2201.04777<br>

**作者**：Bingshu Wang,Jiangbin Zheng,C. L. Philip Chen<br>
**机构**：Northwestern Polytechnical University<br>
**备注**：21 pages, 9 figures, 5 tables. IEEE Transactions on Artificial Intelligence, 2021, early access<br>
**摘要**：2019年冠状病毒病（COVID-19）自爆发以来，继续对世界构成巨大挑战。为了对抗2019冠状病毒疾病，一系列人工智能（AI）技术被开发并应用于真实世界的情景，如安全监测、疾病诊断、感染风险评估、COVID-19 CT扫描的病变分割等。冠状病毒流行病迫使人们戴口罩以对抗病毒的传播，这也给监控戴口罩的人群带来了困难。在这篇论文中，我们主要集中在蒙面人脸检测和相关数据集的人工智能技术。我们综述了最近的进展，首先是对蒙面人脸检测数据集的描述。详细描述和讨论了13个可用数据集。然后，将这些方法大致分为两类：常规方法和基于神经网络的方法。传统的方法通常是通过使用手工特征的boosting算法进行训练，而手工特征只占一小部分。根据加工阶段的数量，基于神经网络的方法进一步分为三部分。详细描述了代表性算法，并简要描述了一些典型技术。最后，我们总结了最近的基准测试结果，讨论了数据集和方法的局限性，并展望了未来的研究方向。据我们所知，这是第一次调查蒙面面部检测方法和数据集。希望我们的调查能为抗击流行病提供一些帮助。

<br><br>

【9】 Survey and Systematization of 3D Object Detection Models and Methods<br>
**标题**：三维目标检测模型和方法的综述与系统化<br>
**链接**：https://arxiv.org/abs/2201.09354<br>

**作者**：Moritz Drobnitzky,Jonas Friederich,Bernhard Egger,Patrick Zschech<br>
**机构**：Technische Universit¨at Dresden, M¨unchner Platz , Dresden, Germany, Mærsk Mc-Kinney Møller Institute, University of Southern Denmark, Campusvej , Odense, Denmark, Friedrich-Alexander-Universit¨at Erlangen-N¨urnberg, Schloßplatz , Erlangen, Germany<br>
**备注**：submitted to CVIU<br>
**摘要**：本文综述了三维目标检测的最新进展，包括从输入数据到实际检测模块的整个流程、超数据表示和特征提取。我们介绍了基本概念，重点调查了过去十年中出现的各种不同方法，并提出了一个系统化的方法，为在方法层面上比较这些方法提供了一个实用的框架。

<br><br>

【10】 3D Object Detection from Images for Autonomous Driving: A Survey<br>
**标题**：自动驾驶图像中三维目标检测的研究进展<br>
**链接**：https://arxiv.org/abs/2202.02980<br>

**作者**：Xinzhu Ma,Wanli Ouyang,Andrea Simonelli,Elisa Ricci<br>
**摘要**：基于图像的三维目标检测是自动驾驶中最基本、最具挑战性的问题之一，近年来受到了工业界和学术界越来越多的关注。得益于深度学习技术的快速发展，基于图像的三维检测技术取得了显著进展。特别是，超过200部作品已经从2015到2021研究了这个问题，涵盖了广泛的理论、算法和应用。然而，到目前为止，还没有最近的调查来收集和整理这些知识。在本文中，我们填补了文献中的这一空白，并对这一新颖且不断增长的研究领域进行了首次全面综述，总结了基于图像的三维检测最常用的管道，并深入分析了其各个组成部分。此外，我们还提出了两种新的分类法，将最先进的方法分为不同的类别，旨在对现有方法进行更系统的审查，并促进与未来作品的公平比较。回顾迄今为止所取得的成就，我们还分析了该领域当前面临的挑战，并讨论了基于图像的三维检测研究的未来方向。

<br><br>

【11】 GAN-generated Faces Detection: A Survey and New Perspectives<br>
**标题**：GaN生成的人脸检测：综述和新视角<br>
**链接**：https://arxiv.org/abs/2202.07145<br>

**作者**：Xin Wang,Hui Guo,Shu Hu,Ming-Ching Chang,Siwei Lyu<br>
**机构**：Keya Medical, Seattle, USA. ,University at Albany, SUNY, USA. ,University at Buffalo, SUNY, USA.<br>
**摘要**：生成性对抗网络（GAN）导致了非常逼真的人脸图像的产生，这些图像被用于伪造社交媒体账户和其他可能产生深远影响的虚假信息。因此，相应的人脸检测技术正在积极开发中，可以检测和暴露这些假人脸。在这项工作中，我们的目的是提供一个全面的审查，最近的进展在氮化镓人脸检测。我们专注于检测由GAN模型生成或合成的人脸图像的方法。我们将现有的检测工作分为四类：（1）基于深度学习的方法，（2）基于物理的方法，（3）基于生理的方法，以及（4）对人类视觉性能的评估和比较。对于每个类别，我们总结关键思想，并将其与方法实现联系起来。我们还讨论了存在的问题，并提出了未来的研究方向。

<br><br>

【12】 A Survey of Visual Sensory Anomaly Detection<br>
**标题**：视觉感觉异常检测技术综述<br>
**链接**：https://arxiv.org/abs/2202.07006<br>

**作者**：Xi Jiang,Guoyang Xie,Jinbao Wang,Yong Liu,Chengjie Wang,Feng Zheng,Yaochu Jin<br>
**机构**：Southern University on Science and Technology, VIP Lab, University of Surrey, NICE Group, Tencent, Youtu Lab, Bielefeld University<br>
**摘要**：视觉感官异常检测（AD）是计算机视觉中的一个基本问题，近年来，随着人工智能的发展，这一问题得到了越来越多的关注。与语义异常检测（语义移位）相比，视觉感知AD检测样本的异常部分（协变量移位）。然而，还没有为计算机视觉界提供全面的综述来总结这一领域。在这项调查中，我们是第一个全面回顾视觉感官广告的人，并根据异常的形式将其分为三个级别。此外，我们根据监管级别对各种异常进行分类。最后，我们总结了挑战，并为这个社区提供了开放的方向。所有资源均可在https://github.com/M-3LAB/awesome-visual-sensory-anomaly-detection.

<br><br>

【13】 A Comprehensive Survey on Video Saliency Detection with Auditory Information: the Audio-visual Consistency Perceptual is the Key!<br>
**标题**：听觉信息视频显著检测综述：视听一致性感知是关键！<br>
**链接**：https://arxiv.org/abs/2206.13390<br>

**作者**：Chenglizhao Chen,Mengke Song,Wenfeng Song,Li Guo,Muwei Jian<br>
**摘要**：视频显著性检测（VSD）旨在快速定位给定视频片段中最具吸引力的对象/事物/模式。现有的VSD相关作品主要依赖视觉系统，而对音频方面的关注较少，而实际上，我们的音频系统是视觉系统最重要的补充部分。此外，视听显著性检测（AVSD）是模仿人类感知机制的最具代表性的研究课题之一，目前还处于起步阶段，现有的调查论文都没有涉及到它，尤其是从显著性检测的角度。因此，本文的最终目的是对视听融合和显著性检测之间的差距进行广泛的回顾。此外，作为本综述的另一个亮点，我们深入了解了可能直接决定AVSD深度模型性能的关键因素，并声称视听一致性程度（AVC）——一个长期被忽视的问题，可以直接影响在执行显著性检测时使用音频使其视觉对应方受益的有效性。此外，为了使AVC问题对未来的关注者更加实用和有价值，我们新为几乎所有现有的公开可用AVSD数据集配备了额外的帧级AVC标签。基于这些升级的数据集，我们进行了广泛的定量评估，以证明AVC在AVSD任务中的重要性。总之，我们的想法和新设置都是一个方便的平台，提供了初步的准备和指导，所有这些都有助于推动未来的工作，进一步促进最先进的（SOTA）性能。

<br><br>

【14】 3D Object Detection for Autonomous Driving: A Review and New Outlooks<br>
**标题**：自动驾驶中的三维目标检测：回顾与新展望<br>
**链接**：https://arxiv.org/abs/2206.09474<br>

**作者**：Jiageng Mao,Shaoshuai Shi,Xiaogang Wang,Hongsheng Li<br>
**机构**：hk) 1 The Chinese University of Hong Kong, China 2 Max Planck Institute for Informatics<br>
**备注**：A survey on 3D object detection for autonomous driving. Project page is at this https URL<br>
**摘要**：近年来，自动驾驶因其在减轻驾驶员负担和提高驾驶安全性方面的潜力而受到越来越多的关注。在现代自动驾驶管道中，感知系统是不可或缺的组成部分，旨在准确估计周围环境的状态，并为预测和规划提供可靠的观测。三维目标检测是感知系统的重要组成部分，它可以智能地预测自主车辆附近关键三维目标的位置、大小和类别。本文综述了自动驾驶三维目标检测的研究进展。首先，我们介绍了三维目标检测的背景，并讨论了本课题面临的挑战。其次，我们从模型和感官输入方面全面综述了三维目标检测的进展，包括基于激光雷达、基于相机和多模式检测方法。我们还深入分析了每类方法的潜力和挑战。此外，我们还系统地研究了三维目标检测在驱动系统中的应用。最后，我们对三维目标检测方法进行了性能分析，并进一步总结了近年来的研究趋势，展望了该领域未来的发展方向。

<br><br>

【15】 A Survey of Detection Methods for Die Attachment and Wire Bonding Defects in Integrated Circuit Manufacturing<br>
**标题**：集成电路制造中芯片连接和引线键合缺陷检测方法综述<br>
**链接**：https://arxiv.org/abs/2206.07481<br>

**作者**：Lamia Alam,Nasser Kehtarnavaz<br>
**机构**：Department of Electrical and Computer Engineering, University of Texas at Dallas, Richardson, TX<br>
**备注**：13 pages, 9 figures, 8 tables<br>
**摘要**：缺陷检测在集成电路制造过程中起着至关重要的作用。芯片连接和引线键合是制造过程中的两个步骤，它们决定了IC中电源和信号传输的质量和可靠性。本文介绍了基于不同传感模式（包括光学、放射学、声学和红外热成像）检测这些缺陷的方法的综述或文献综述。本次调查对所使用的检测方法进行了讨论。传统的和深入的学习方法检测芯片连接和引线键合缺陷被认为是挑战和未来的研究方向。

<br><br>

【16】 A Survey of Deep Fake Detection for Trial Courts<br>
**标题**：浅谈庭审中的深度假冒检测<br>
**链接**：https://arxiv.org/abs/2205.15792<br>

**作者**：Naciye Celebi,Qingzhong Liu,Muhammed Karatoprak<br>
**机构**：Department of Computer Science, Sam Houston State University, Huntsville, TX, USA, Department LLM in US Law, The University. Of Houston, Houston, TX, USA<br>
**备注**：12 Pages, 1 Table<br>
**摘要**：近年来，由于先进的图像编辑工具的发展，图像处理得到了快速发展。最近，使用神经网络生成的虚假图像和视频激增，这就是DeepFake。DeepFake算法可以创建人类无法区分的虚假图像和视频。（GANs）已被广泛用于创建逼真的图像，而无需访问原始图像。因此，检测虚假视频以避免虚假信息的传播变得至关重要。本文综述了迄今为止文献中用于检测深度伪造的方法和数据集。我们介绍了与DeepFake技术相关的广泛讨论和研究趋势。

<br><br>

【17】 Continual Object Detection: A review of definitions, strategies, and challenges<br>
**标题**：连续目标检测：定义、策略和挑战的回顾<br>
**链接**：https://arxiv.org/abs/2205.15445<br>

**作者**：Angelo G. Menezes,Gustavo de Moura,Cézanne Alves,André C. P. L. F. de Carvalho<br>
**机构**：Institute of Mathematics and Computer Sciences, University of São Paulo, Av. Trab. São Carlense, - Centro, São Carlos,-, São Paulo, Brazil, Eldorado Research Institute, Av. Alan Turing, Cidade Universitária, Campinas,-, São Paulo, Brazil<br>
**摘要**：持续学习领域研究的是学习连续任务的能力，而不会失去先前学习的成绩。其重点主要是增量分类任务。我们相信，由于连续目标检测在机器人和自动车辆中的广泛应用，其研究值得更多的关注。由于出现了当时未知的类实例，但可以作为要学习的新类出现在后续任务中，从而导致注释缺失和与背景标签冲突，因此该场景比传统分类更复杂。在这篇综述中，我们分析了当前提出的解决类增量目标检测问题的策略。我们的主要贡献是：（1）对提出传统增量目标检测方案的方法进行了简短而系统的回顾；（2） 对现有方法进行综合评估，使用新指标以标准方式量化每种技术的稳定性和可塑性；（3） 概述了连续目标检测的当前趋势，并讨论了未来可能的研究方向。

<br><br>

<a name="transformer"/>

# **Transformer**

【1】 Transformers in 3D Point Clouds: A Survey<br>
**标题**：三维点云中的变形器：综述<br>
**链接**：https://arxiv.org/abs/2205.07417<br>

**作者**：Dening Lu,Qian Xie,Mingqiang Wei,Linlin Xu,Jonathan Li<br>
**机构**：Nanjing University of Aeronautics and Astronautics, Jonathan Li is with the Department of Geography and EnvironmentalManagement, University of Waterloo<br>
**备注**：22 pages, 5 figures, 5 tables<br>
**摘要**：近年来，Transformer模型已被证明具有显著的远程依赖建模能力。他们在自然语言处理（NLP）和图像处理方面都取得了令人满意的结果。这一重大成就激发了研究人员对3D点云处理的极大兴趣，并将其应用于各种3D任务。由于其固有的排列不变性和强大的全局特征学习能力，3D Transformers非常适合于点云处理和分析。与最先进的无Transformer算法相比，它们实现了具有竞争力甚至更好的性能。本调查旨在全面概述为各种任务（例如点云分类、分割、对象检测等）设计的3D转换器。我们首先介绍通用Transformer的基本部件，并简要介绍其在二维和三维领域的应用。然后，我们为方法分类提出了三种不同的分类法（即基于Transformer实现的分类法、基于数据表示的分类法和基于任务的分类法），这使我们能够从多个角度分析涉及的方法。此外，我们还研究了为提高性能而设计的3D自我注意机制变体。为了证明3D Transformers的优越性，我们比较了基于Transformer的算法在点云分类、分割和目标检测方面的性能。最后指出了未来三个可能的研究方向，以期为三维Transformer的发展提供有益的参考。

<br><br>

【2】 Image Captioning In the Transformer Age<br>
**标题**：Transformer时代的图像字幕<br>
**链接**：https://arxiv.org/abs/2204.07374<br>

**作者**：Yang Xu,Li Li,Haiyang Xu,Songfang Huang,Fei Huang,Jianfei Cai<br>
**机构**：School of Computer Science and Engineering of Southeast University, China, Bell Honors School, Nanjing University of Posts and Telecommunications, Nanjing, China, Alibaba Group, Department of Data Science and AI, Monash University<br>
**备注**：8pages,2 figures<br>
**摘要**：图像字幕（IC）通过将各种技术整合到CNN-RNN编解码器体系结构中，取得了惊人的发展。然而，由于CNN和RNN不共享基本的网络组件，这样的异构管道很难进行端到端的训练，视觉编码器将无法从字幕监控中学习任何东西。这一缺陷促使研究人员开发了一种促进端到端训练的同质体系结构，Transformer是一种完美的体系结构，已证明其在视觉和语言领域都具有巨大潜力，因此可以用作IC管道中视觉编码器和语言解码器的基本组件。同时，自监督学习释放了Transformer体系结构的力量，即预先训练的大规模Transformer可以推广到包括IC在内的各种任务。这些大规模模型的成功似乎削弱了单一IC任务的重要性。然而，通过分析IC与一些流行的自我监督学习范式之间的联系，我们证明了IC在这个时代仍然有其特殊的意义。由于篇幅限制，我们在这篇简短的调查中只参考了非常重要的论文，更多相关的工作可以在https://github.com/SjokerLily/awesome-image-captioning.

<br><br>

【3】 Data and Physics Driven Learning Models for Fast MRI -- Fundamentals and Methodologies from CNN, GAN to Attention and Transformers<br>
**标题**：用于快速磁共振成像的数据和物理驱动的学习模型--从CNN、GAN到注意力和转换器的基本原理和方法<br>
**链接**：https://arxiv.org/abs/2204.01706<br>

**作者**：Jiahao Huang,Yingying Fang,Yang Nan,Huanjun Wu,Yinzhe Wu,Zhifan Gao,Yang Li,Zidong Wang,Pietro Lio,Daniel Rueckert,Yonina C. Eldar,Guang Yang<br>
**备注**：14 pages, 3 figures, submitted to IEEE SPM<br>
**摘要**：研究表明，在医学图像分析的下游任务中使用数据驱动的深度学习模型毫无疑问，例如解剖分割和病变检测、疾病诊断和预后以及治疗计划。然而，当上游成像没有正确进行（使用人工制品）时，深度学习模型并不是医学图像分析的主要补救措施。这一点在MRI研究中得到了证实，在MRI研究中，扫描速度通常较慢，容易出现运动伪影，信噪比相对较低，空间和/或时间分辨率较差。最近的研究已经见证了推动快速MRI的深度学习技术的发展。本文旨在（1）介绍基于深度学习的快速MRI数据驱动技术，包括卷积神经网络和基于生成对抗网络的方法；（2）综述用于加速MRI重建的基于注意和Transformer的模型；（3）详细研究MRI加速的物理和数据驱动耦合模型。最后，我们将通过一些临床应用进行演示，解释数据协调的重要性以及在多中心和多扫描仪研究中此类快速MRI技术的可解释模型，并讨论当前研究中的常见缺陷以及对未来研究方向的建议。

<br><br>

【4】 Vision Transformers in Medical Computer Vision -- A Contemplative Retrospection<br>
**标题**：医学计算机视觉中的视觉转换器--反思<br>
**链接**：https://arxiv.org/abs/2203.15269<br>

**作者**：Arshi Parvaiz,Muhammad Anwaar Khalid,Rukhsana Zafar,Huma Ameer,Muhammad Ali,Muhammad Moazam Fraz<br>
**机构**：School of Electrical Engineering and Computer Science, National University of Sciences and Technology (NUST), Islamabad, Pakistan, A R T I C L E I N F O<br>
**摘要**：计算机视觉领域最近的升级巩固了一系列算法，这些算法具有解开图像中包含的信息的巨大潜力。这些计算机视觉算法正在医学图像分析中实践，并正在改变对成像数据的感知和解释。在这些算法中，视觉变换器逐渐发展成为计算机视觉领域中最现代、最主要的体系结构之一。许多研究人员都在大量利用这些技术进行新的和以前的实验。在本文中，我们研究了视觉转换器和医学图像的交叉点，并概述了各种基于ViTs的框架，不同的研究人员正在使用这些框架来破译医学计算机视觉中的障碍。我们调查了视觉变换器在医学计算机视觉不同领域的应用，例如基于图像的疾病分类、解剖结构分割、配准、基于区域的病变检测、字幕、报告生成、使用多种医学成像模式进行重建，这些模式大大有助于医学诊断和治疗过程。除此之外，我们还揭开了医学计算机视觉中使用的几种成像模式的神秘面纱。此外，为了获得更多的洞察力和更深的理解，本文还简要解释了Transformer的自我注意机制。最后，我们还以讨论的形式介绍了可用的数据集、采用的方法、它们的性能度量、挑战和解决方案。我们希望这篇综述文章将为医学计算机视觉的研究人员打开未来的方向。

<br><br>

【5】 Video Transformers: A Survey<br>
**标题**：视频Transformer：综述<br>
**链接**：https://arxiv.org/abs/2201.05991<br>

**作者**：Javier Selva,Anders S. Johansen,Sergio Escalera,Kamal Nasrollahi,Thomas B. Moeslund,Albert Clapés<br>
**机构**：Universitat de Barcelona ,Aalborg University ,Computer Vision Center ,Milestone Systems<br>
**摘要**：Transformer模型已经显示了对远程交互建模的巨大成功。然而，它们与输入长度成二次比例，并且缺少电感偏置。在处理高维视频时，这些限制可能会进一步加剧。视频的正确建模可以从几秒钟到几小时不等，需要处理远程交互。这使得Transformers成为解决视频相关任务的一个很有前途的工具，但需要进行一些调整。虽然之前有研究视觉任务中Transformer的进展的工作，但没有一项专注于视频特定设计的深入分析。在这项调查中，我们分析和总结了Transformer适应模型视频数据的主要贡献和趋势。具体来说，我们深入研究了视频是如何嵌入和标记的，发现广泛使用大型CNN主干来降低维度，并且主要使用补丁和帧作为标记。此外，我们还研究了如何调整Transformer层来处理更长的序列，通常是通过减少单注意操作中的令牌数。此外，我们还分析了用于训练视频Transformer的自监督损耗，到目前为止，这些损耗主要局限于对比方法。最后，我们探讨了其他模式如何与视频相结合，并对视频转换器最常见的基准（即动作分类）进行了性能比较，发现它们在相同的失败率和无显著参数增加的情况下优于3D CNN。

<br><br>

【6】 Transformers in Medical Imaging: A Survey<br>
**标题**：医学影像中的Transformer：综述<br>
**链接**：https://arxiv.org/abs/2201.09873<br>

**作者**：Fahad Shamshad,Salman Khan,Syed Waqas Zamir,Muhammad Haris Khan,Munawar Hayat,Fahad Shahbaz Khan,Huazhu Fu<br>
**机构**： Monash University, Australian National University<br>
**备注**：41 pages, \url{this https URL}<br>
**摘要**：在自然语言任务上取得了前所未有的成功之后，Transformers已经成功地应用于几个计算机视觉问题，取得了最先进的结果，并促使研究人员重新考虑卷积神经网络（CNN）作为{fact}算子的优越性。利用计算机视觉的这些进步，医学成像领域也见证了与具有局部感受野的CNN相比，人们对能够捕捉全球环境的Transformer越来越感兴趣。受这一转变的启发，在本次调查中，我们试图全面回顾Transformer在医学成像中的应用，涵盖各个方面，从最近提出的建筑设计到尚未解决的问题。具体来说，我们调查了Transformer在医学图像分割、检测、分类、重建、合成、配准、临床报告生成和其他任务中的使用。特别是，对于这些应用程序中的每一个，我们都开发了分类法，确定了特定于应用程序的挑战，并提供了解决这些挑战的见解，并强调了最近的趋势。此外，我们还对该领域的整体现状进行了批判性讨论，包括确定关键挑战、公开问题，并概述了有希望的未来方向。我们希望这项调查将激发社区的进一步兴趣，并为研究人员提供有关Transformer模型在医学成像中应用的最新参考。最后，为了应对这一领域的快速发展，我们打算定期更新相关的最新论文及其开源实现，网址为\url{https://github.com/fahadshamshad/awesome-transformers-in-medical-imaging}.

<br><br>

【7】 Recent Advances in Vision Transformer: A Survey and Outlook of Recent Work<br>
**标题**：视觉转换器的最新进展：近期工作的回顾与展望<br>
**链接**：https://arxiv.org/abs/2203.01536<br>

**作者**：Khawar Islam<br>
**备注**：Added AAAI 2022 methods and working on ICLR 2022 methods<br>
**摘要**：与卷积神经网络（CNN）相比，视觉变换器（VIT）在各种视觉任务中正变得越来越流行和主导。作为计算机视觉中一项要求很高的技术，ViTs已经成功地解决了各种视觉问题，同时专注于长期关系。本文首先介绍了自我注意机制的基本概念和背景。接下来，我们将从优势和劣势、计算成本以及训练和测试数据集等方面全面概述最近表现最好的ViT方法。我们在流行的基准数据集上彻底比较了各种ViT算法和最具代表性的CNN方法的性能。最后，我们通过深入的观察探讨了一些局限性，并提出了进一步的研究方向。项目页面和论文集可在https://github.com/khawar512/ViT-Survey

<br><br>

【8】 Multimodal Learning with Transformers: A Survey<br>
**标题**：利用Transformer进行多模式学习：综述<br>
**链接**：https://arxiv.org/abs/2206.06488<br>

**作者**：Peng Xu,Xiatian Zhu,David A. Clifton<br>
**机构**： Clifton are with Department of EngineeringScience, University of Oxford, •Xiatian Zhu is with Surrey Institute for People-Centred ArtificialIntelligence, University of Surrey<br>
**摘要**：Transformer是一个很有前途的神经网络学习者，在各种机器学习任务中都取得了巨大的成功。由于近年来多模式应用和大数据的流行，基于Transformer的多模式学习已成为人工智能研究的一个热门话题。本文综述了面向多模态数据的Transformer技术。本次调查的主要内容包括：（1）多模式学习、Transformer生态系统和多模式大数据时代的背景，（2）从几何拓扑角度对Vanilla Transformer、Vision Transformer和多模式Transformers进行理论回顾，（3）通过两个重要范式对多模式Transformer的应用进行回顾，即：。，对于多式联运预训练和特定的多式联运任务，（4）总结多式联运Transformer模型和应用共享的共同挑战和设计，（5）讨论社区的开放问题和潜在研究方向。

<br><br>

<a name="医学相关"/>

# **医学相关**

【1】 Development of Diabetic Foot Ulcer Datasets: An Overview<br>
**标题**：糖尿病足溃疡数据集的研究进展<br>
**链接**：https://arxiv.org/abs/2201.00163<br>

**作者**：Moi Hoon Yap,Connah Kendrick,Neil D. Reeves,Manu Goyal,Joseph M. Pappachan,Bill Cassidy<br>
**机构**： Centre for Advanced Computational Science, Department of Computing and, Mathematics, Manchester Metropolitan University, Manchester M,GD, United, Lancashire Teaching Hospitals NHS Trust, Preston, PR,HT, United Kingdom<br>
**备注**：Preprint (author copy) to be published in MICCAI DFUC2021 Proceedings<br>
**摘要**：本文提供了在过去十年中糖尿病足溃疡数据集开发的概念基础和程序，并用时间线来证明进展。我们对足部照片的数据采集方法进行了调查，概述了开发私人和公共数据集的研究，相关的计算机视觉任务（检测、分割和分类），糖尿病足溃疡的挑战以及数据集的未来发展方向。我们按国家和年份报告数据集用户的分布情况。我们的目标是分享我们在数据集开发中遇到的技术挑战和良好实践，并激励其他研究人员参与该领域的数据共享。

<br><br>

【2】 Deep Learning Applications for Lung Cancer Diagnosis: A systematic review<br>
**标题**：深度学习在肺癌诊断中的应用：系统综述<br>
**链接**：https://arxiv.org/abs/2201.00227<br>

**作者**：Hesamoddin Hosseini,Reza Monsefi,Shabnam Shadroo<br>
**机构**：a Department of Computer Engineering, Ferdowsi University of Mashhad, Mashhad, Iran, b Department of Software Engineering, Islamic Azad University, Mashhad Branch, Mashhad, Iran, Corresponding Author<br>
**摘要**：肺癌是近年来最常见的疾病之一。根据这一领域的研究，美国每年发现的病例超过200000例。肺细胞不受控制的增殖和生长导致恶性肿瘤的形成。近年来，深度学习算法，特别是卷积神经网络（CNN），已经成为自动诊断疾病的一种优越方法。本文的目的是回顾导致早期肺癌诊断的不同准确性和敏感性的不同模型，并帮助该领域的医生和研究人员。这项工作的主要目的是在深入学习的基础上确定肺癌中存在的挑战。本研究系统梳理了常规制图与文献综述相结合的方法，对2016至2021年间的32篇会议论文和期刊论文进行了综述。在分析和审查这些条款之后，这些条款中提出的问题正在得到回答。由于对相关文章的全面回顾和系统的总结，本研究优于该领域的其他综述文章。

<br><br>

【3】 A Survey of Left Atrial Appendage Segmentation and Analysis in 3D and 4D Medical Images<br>
**标题**：3D和4D医学图像中左心耳分割与分析的研究进展<br>
**链接**：https://arxiv.org/abs/2205.06486<br>

**作者**：Hrvoje Leventić,Marin Benčević,Danilo Babin,Marija Habijan,Irena Galić<br>
**机构**：Computer, Science and Information Technologies, Osijek, Croatia, TELIN-IPI, Ghent University – imec, Belgium<br>
**摘要**：心房颤动（AF）是一种心血管疾病，被认为是中风的主要危险因素之一。房颤引起的大多数中风是由左心耳（LAA）的血栓引起的。LAA封堵术是降低中风风险的有效方法。使用术前成像和分析来规划手术已经显示出好处。分析通常通过在2D切片上手动分割附属物来完成。自动LAA分割方法可以节省专家的时间，并提供深入的3D可视化和精确的自动测量，以帮助医疗程序。已经提出了几种半自动和全自动分割附属物的方法。本文综述了3D和4D医学图像的自动LAA分割方法，包括CT、MRI和超声心动图图像。我们将方法分为启发式方法和基于模型的方法，以及半自动和全自动方法。我们总结和比较了提出的方法，评估了它们的有效性，并提出了该领域当前面临的挑战和克服这些挑战的方法。

<br><br>

【4】 Explainable Deep Learning Methods in Medical Diagnosis: A Survey<br>
**标题**：医学诊断中可解释的深度学习方法研究综述<br>
**链接**：https://arxiv.org/abs/2205.04766<br>

**作者**：Cristiano Patrício,João C. Neves,Luís F. Teixeira<br>
**机构**： University of Beira Interior, University of Porto<br>
**摘要**：深度学习的巨大成功促使人们对其在医学诊断中的应用产生了兴趣。即使是最先进的深度学习模型在不同类型医疗数据的分类上也达到了人类水平的准确性，但这些模型在临床工作流程中几乎没有被采用，主要是因为它们缺乏可解释性。深度学习模型的黑匣子性使得人们需要设计策略来解释这些模型的决策过程，从而产生了可解释人工智能（XAI）这一主题。在此背景下，我们对XAI在医学诊断中的应用进行了全面的调查，包括视觉、文本和基于实例的解释方法。此外，这项工作回顾了现有的医学影像数据集和现有的评估解释质量的指标。作为对大多数现有调查的补充，我们对一组基于报告生成的方法进行了性能比较。最后，还讨论了将XAI应用于医学成像的主要挑战。

<br><br>

【5】 A survey on attention mechanisms for medical applications: are we moving towards better algorithms?<br>
**标题**：医学应用中注意力机制的调查：我们正在朝着更好的算法发展吗？<br>
**链接**：https://arxiv.org/abs/2204.12406<br>

**作者**：Tiago Gonçalves,Isabel Rio-Torto,Luís F. Teixeira,Jaime S. Cardoso<br>
**备注**：Pre-print submitted to Nature Scientific Reports<br>
**摘要**：在计算机视觉和自然语言处理的深度学习算法中，注意力机制的日益普及使得这些模型对其他研究领域具有吸引力。在医疗保健领域，迫切需要能够改善临床医生和患者日常生活的工具。自然，基于注意的算法在医学应用中的应用进展顺利。然而，由于医疗保健是一个依赖于高风险决策的领域，科学界必须考虑这些高性能算法是否适合医疗应用的需要。基于这一座右铭，本文广泛回顾了机器学习（包括Transformer）中注意机制在几种医学应用中的使用。这项工作有别于前人，它通过对三种不同使用案例的医学图像分类实验案例研究，对文献中提出的注意机制的主张和潜力进行了批判性分析。这些实验的重点是将注意力机制的过程整合到已建立的深度学习架构中，分析它们的预测能力，以及通过事后解释方法生成的显著性图的视觉评估。本文最后对有关注意机制的文献中提出的主张和潜力进行了批判性分析，并提出了可能受益于这些框架的未来医学应用研究方向。

<br><br>

【6】 A Survey on Training Challenges in Generative Adversarial Networks for Biomedical Image Analysis<br>
**标题**：生物医学图像分析生成对抗网络训练挑战研究综述<br>
**链接**：https://arxiv.org/abs/2201.07646<br>

**作者**：Muhammad Muneeb Saad,Ruairi O'Reilly,Mubashir Husain Rehmani<br>
**机构**：Department of Computer Science, Munster Technological University (MTU), Ireland<br>
**摘要**：在生物医学图像分析中，深度学习方法的适用性直接受到可用图像数据量的影响。这是因为深度学习模型需要大的图像数据集来提供高水平的性能。生成性对抗网络（GAN）已被广泛用于通过生成合成生物医学图像来解决数据限制。GANs由两个模型组成。生成器，一个学习如何根据接收到的反馈生成合成图像的模型。鉴别器，一种将图像分类为合成图像或真实图像并向生成器提供反馈的模型。在整个训练过程中，GAN可能会遇到一些阻碍生成合适合成图像的技术挑战。首先，模式崩溃问题，即生成器要么生成相同的图像，要么从不同的输入特征生成统一的图像。第二，梯度下降优化器无法达到纳什均衡的非收敛问题。第三，消失梯度问题，即由于鉴别器实现了最佳分类性能，导致没有向生成器提供有意义的反馈，从而出现不稳定的训练行为。这些问题导致合成图像的生成变得模糊、不切实际，且不太多样化。迄今为止，还没有任何调查文章概述这些技术挑战在生物医学图像领域的影响。这项工作提出了一个审查和分类的基础上解决方案的训练问题的肝在生物医学成像领域。这项调查突出了重要的挑战，并概述了未来在生物医学成像领域中有关GANs训练的研究方向。

<br><br>

【7】 A Survey of Breast Cancer Screening Techniques: Thermography and Electrical Impedance Tomography<br>
**标题**：乳腺癌筛查技术综述：热成像和电阻抗断层扫描<br>
**链接**：https://arxiv.org/abs/2202.03737<br>

**作者**：Juan Zuluaga-Gomez,N. Zerhouni,Z. Al Masry,C. Devalland,C. Varnier<br>
**机构**：FEMTO-ST institute, Univ. Bourgogne Franche-Comt´e, CNRS, ENSMM, Besan¸con, France; bElectrical Engineering Department, University of Oviedo, Gijon, Spain;, Universidad Autonoma del Caribe, Barranquilla, Colombia; dPathology Department<br>
**摘要**：乳腺癌是一种威胁许多女性生命的疾病，因此，早期准确的检测对降低死亡率起着关键作用。乳房X光摄影是乳腺癌筛查的参考技术；然而，由于经济、社会和文化问题，许多国家仍然无法获得乳房X光检查。计算工具、红外相机和生物阻抗量化设备的最新进展允许开发并行技术，如热成像、红外成像和电阻抗层析成像，这些技术更快、更可靠、更便宜。在过去几十年中，这些被认为是乳腺癌诊断的补充程序，许多研究得出结论，假阳性和假阴性率大大降低。这项工作旨在回顾关于上述三种技术的最新突破，它们描述了混合几种计算技能以获得更好的全局性能的好处。此外，我们还比较了几种用于乳腺癌诊断的机器学习技术，从逻辑回归、决策树和随机森林到人工、深层和卷积神经网络。最后，对三维乳房模拟、预处理技术、研究领域的生物医学设备、肿瘤位置和大小的预测提出了几点建议。

<br><br>

【8】 Deep Learning for Computational Cytology: A Survey<br>
**标题**：计算细胞学的深度学习：综述<br>
**链接**：https://arxiv.org/abs/2202.05126<br>

**作者**：Hao Jiang,Yanning Zhou,Yi Lin,Ronald CK Chan,Jiang Liu,Hao Chen<br>
**机构**：Department of Computer Science and Engineering, The Hong Kong University of Science and Technology, Hong Kong, China, Department of Computer Science and Engineering, The Chinese University of Hong Kong, Hong Kong, China<br>
**摘要**：在医学图像计算领域，计算细胞学是一个关键的、快速发展的、但具有挑战性的课题，它通过计算机辅助技术对数字化的细胞学图像进行分析，用于癌症筛查。近年来，越来越多的深度学习（DL）算法在医学图像分析方面取得了重大进展，从而推动了细胞学研究的发表。为了研究这种先进的方法和广泛的应用，我们在本文中调查了120多篇基于DL的细胞学图像分析的出版物。我们首先介绍了各种深度学习方法，包括完全监督、弱监督、无监督和迁移学习。然后，我们系统地总结了公共数据集、评估指标、多种细胞学图像分析应用，包括分类、检测、分割等相关任务。最后，我们讨论了计算细胞学目前面临的挑战和潜在的研究方向。

<br><br>

【9】 A Survey of Deep Learning Techniques for the Analysis of COVID-19 and their usability for Detecting Omicron<br>
**标题**：深度学习技术在冠状病毒分析中的应用及其在检测Omicron中的应用<br>
**链接**：https://arxiv.org/abs/2202.06372<br>

**作者**：Asifullah Khan,Saddam Hussain Khan,Mahrukh Saif,Asiya Batool,Anabia Sohail,Muhammad Waleed Khan<br>
**摘要**：2019年12月爆发的新冠病毒（COVID-19）已成为全世界人类的持续威胁，造成了一场健康危机，感染了数百万人的生命，并破坏了全球经济。事实证明，深度学习（DL）技术有助于及时分析和描绘放射图像中的感染区域。本文对DL技术进行了深入的调查，并根据诊断策略和学习方法绘制了分类法。DL 2019冠状病毒疾病的分类和分割，以及多阶段的图像编码和区域分析。2019冠状病毒疾病的影像学检查方法包括：预先训练和定制的卷积神经网络结构，用于检测COVID-19感染；X射线和计算机断层扫描（CT）。此外，还讨论了在大流行中开发诊断技术、跨平台互操作性和检查成像模式所面临的挑战，以及这些技术中使用的方法和性能度量。2019冠状病毒疾病的研究提供了一个深入的研究领域，从而进一步促进了基于DL的诊断工具的设计研究，以有效地处理COVID-19的新变体和新的挑战。

<br><br>

【10】 A Survey of Semen Quality Evaluation in Microscopic Videos Using Computer Assisted Sperm Analysis<br>
**标题**：计算机辅助精子分析在显微视频精液质量评价中的应用<br>
**链接**：https://arxiv.org/abs/2202.07820<br>

**作者**：Wenwei Zhao,Pingli Ma,Chen Li,Xiaoning Bu,Shuojia Zou,Tao Jang,Marcin Grzegorzek<br>
**机构**： Microscopic Image and Medical Image Analysis Group, MBIE College, Northeastern University, Shenyang, China, University of Washington (Seattle Campus), Seattle, US, School of Control Engineering, Chengdu University of Information Technology, Chengdu, PR China<br>
**摘要**：计算机辅助精子分析（CASA）在男性生殖健康诊断和不孕症治疗中起着至关重要的作用。近年来，随着计算机工业的发展，人们提出了许多精确的算法。在这些新算法的帮助下，CASA有可能获得更快、更高质量的结果。由于图像处理是CASA的技术基础，包括预处理、特征提取、目标检测和跟踪，这些方法是处理CASA的重要技术步骤。本文全面介绍和分析了近30年（1988年以来）计算机辅助精子分析方法的相关工作。为了便于理解，按照精子分析的一般步骤顺序分析所涉及的方法。换句话说，首先分析与精子检测（定位）相关的方法，然后分析精子跟踪的方法。此外，我们还对CASA的现状和未来进行了分析和展望。根据我们的工作，解释了本综述中提到的方法在精子显微视频中应用的可行性。此外，受这项调查的启发，显微镜视频中存在的目标检测和跟踪挑战有可能得到解决。

<br><br>

【11】 An overview of deep learning in medical imaging<br>
**标题**：医学影像学中的深度学习研究综述<br>
**链接**：https://arxiv.org/abs/2202.08546<br>

**作者**：Imran Ul Haq<br>
**机构**：College of Information Sciences and Technology, Northwest University, Xi’an , China<br>
**摘要**：在最近的十年里，机器学习（ML）受到了广泛的关注。这一成功始于2012年，当时一个ML模型在ImageNet分类中取得了显著的胜利，这是世界上最著名的计算机视觉竞赛。该模型是一种称为深度学习（DL）的卷积神经系统（CNN）。从那时起，研究人员开始有效地参与DL发展最快的研究领域。如今，DL系统是最前沿的ML系统，涵盖了从人类语言处理到视频分析等多个学科，广泛应用于学术界和企业部门。最近的进步可以给医学领域带来巨大的进步。改进和创新数据处理、图像分析方法，可以显著提高诊断技术和医疗服务水平。本文简要回顾了用于医学成像的DL领域的最新发展和相关问题。本综述的主要目的有四个：（i）通过讨论不同的DL模型，为DL提供一个简短的序言，（ii）回顾DL在医学图像分析（分类、检测、分割和配准）中的使用，（iii）回顾DL在医学成像中的七个主要应用领域，（iv）通过提供一些有用信息资产的链接，例如免费提供的DL代码、公共数据集表7和医学影像竞赛资源表8，为那些热衷于在临床影像学中增加DL研究领域的人提供一个初始阶段，并通过概述明显的持续困难来结束我们的调查，DL在医学领域的经验教训和未来。

<br><br>

【12】 Automated image analysis in large-scale cellular electron microscopy: A literature survey<br>
**标题**：大规模细胞电子显微镜中的自动图像分析：文献综述<br>
**链接**：https://arxiv.org/abs/2206.07171<br>

**作者**：Anusha Aswatha,Ahmad Alsahaf,Ben N. G. Giepmans,George Azzopardi<br>
**机构**：Azzopardia, Bernoulli Institute of Mathematics, Computer Science and Artificial Intelligence, University Groningen, Groningen, The Netherlands, Dept. Biomedical Sciences of Cells and Systems, University Groningen, University Medical<br>
**摘要**：使用（半）自动显微镜生成的大规模电子显微镜（EM）数据集正在成为EM的标准。鉴于数据量巨大，手动分析所有数据是不可行的，因此自动分析至关重要。自动化分析的主要挑战包括分析和解释生物医学图像所需的注释，以及实现高通量。在此，我们回顾了自动化计算机技术的现状以及细胞EM结构分析面临的主要挑战。讨论了过去五年中为自动生物医学图像分析开发的先进计算机视觉、深度学习和软件工具，以及EM数据的注释、分割和可扩展性。自动图像采集和分析的集成将允许以纳米分辨率对毫米级数据集进行高通量分析。

<br><br>

【13】 Applications of Generative Adversarial Networks in Neuroimaging and Clinical Neuroscience<br>
**标题**：生成性对抗网络在神经影像和临床神经科学中的应用<br>
**链接**：https://arxiv.org/abs/2206.07081<br>

**作者**：Rongguang Wang,Vishnu Bashyam,Zhijian Yang,Fanyang Yu,Vasiliki Tassopoulou,Lasya P. Sreepada,Sai Spandana Chintapalli,Dushyant Sahoo,Ioanna Skampardoni,Konstantina Nikita,Ahmed Abdulkadir,Junhao Wen,Christos Davatzikos<br>
**机构**： University of Pennsylvania, USA 2 School of Electrical and Computer Engineering, National Technical University of Athens, Greece 3 University Hospital of Old Age Psychiatry and Psychotherapy, University of Bern<br>
**摘要**：生成性对抗网络（GAN）是一种强大的深度学习模型，已成功应用于许多领域。它们属于一个更广泛的称为生成方法的家族，该家族通过从真实示例中学习样本分布来使用概率模型生成新数据。在临床方面，与传统的生成方法相比，GANs在捕捉空间复杂、非线性和潜在细微疾病影响方面表现出更强的能力。本文综述了GANs在各种神经疾病影像学研究中的应用，包括阿尔茨海默病、脑肿瘤、脑老化和多发性硬化症。我们为每个应用提供了各种GAN方法的直观解释，并进一步讨论了在神经成像中利用GAN的主要挑战、开放性问题和有希望的未来方向。我们旨在通过强调如何利用GANs来支持临床决策，并有助于更好地理解大脑疾病的结构和功能模式，从而弥合高级深度学习方法与神经病学研究之间的差距。

<br><br>

【14】 Reinforcement Learning in Medical Image Analysis: Concepts, Applications, Challenges, and Future Directions<br>
**标题**：强化学习在医学图像分析中的应用：概念、应用、挑战和未来方向<br>
**链接**：https://arxiv.org/abs/2206.14302<br>

**作者**：Mingzhe Hu,Jiahan Zhang,Luke Matkovic,Tian Liu,Xiaofeng Yang<br>
**机构**：Department of Computer Science and Informatics, Emory University, GA, Atlanta, USA, Department of Radiation Oncology, Winship Cancer Institute, School of Medicine<br>
**摘要**：动机：医学图像分析包括帮助医生对病变或解剖结构进行定性和定量分析，显著提高诊断和预后的准确性和可靠性。传统上，这些任务是由医生或医学物理学家完成的，导致两个主要问题：（i）效率低下；（ii）个人经验有偏见。在过去的十年中，许多机器学习方法被应用于加速和自动化图像分析过程。与监督和非监督学习模型的大量部署相比，在医学图像分析中使用强化学习的尝试很少。这篇综述文章可以作为相关研究的垫脚石。意义：根据我们的观察，虽然强化学习近年来逐渐获得了发展势头，但医学分析领域的许多研究人员发现很难理解并在临床上应用。一个原因是缺乏针对缺乏专业计算机科学背景的读者的组织良好的评论文章。本文不是提供医学图像分析中所有强化学习模型的综合列表，而是帮助读者了解如何将其医学图像分析研究作为强化学习问题来制定和解决。方法与结果：我们从Google Scholar和PubMed中选择已发表的文章。考虑到相关文章的稀缺性，我们还包括一些优秀的最新预印本。论文根据图像分析任务的类型进行了仔细的审查和分类。我们首先回顾了强化学习的基本概念和流行模型。然后探讨了强化学习模型在地标检测中的应用。最后，我们通过讨论所回顾的强化学习方法的局限性和可能的改进来总结文章。

<br><br>

<a name="多模态"/>

# **多模态**

【1】 Deep Learning in Multimodal Remote Sensing Data Fusion: A Comprehensive Review<br>
**标题**：深度学习在多模式遥感数据融合中的应用综述<br>
**链接**：https://arxiv.org/abs/2205.01380<br>

**作者**：Jiaxin Li,Danfeng Hong,Lianru Gao,Jing Yao,Ke Zheng,Bing Zhang,Jocelyn Chanussot<br>
**机构**：Chanussote,b, Key Laboratory of Computational Optical Imaging Technology, Aerospace Information Research Institute, Aerospace Information Research Institute, Chinese Academy of Sciences, Beijing , China;<br>
**摘要**：随着遥感（RS）技术的飞速发展，大量具有相当复杂异质性的地球观测（EO）数据已经面世，这使研究人员有机会以全新的方式处理当前的地球科学应用。近年来，随着地球观测数据的联合利用，多模态遥感数据融合的研究取得了巨大进展，但由于缺乏对这些高度异构数据的综合分析和解释能力，这些发展起来的传统算法不可避免地遇到了性能瓶颈。因此，这一不可忽视的限制进一步引发了对具有强大处理能力的替代工具的强烈需求。深度学习（Deep learning，DL）作为一种前沿技术，由于其在数据表示和重建方面的强大能力，在许多计算机视觉任务中取得了显著的突破。自然，它已经成功地应用于多模态遥感数据融合领域，与传统方法相比有了很大的改进。本综述旨在对基于DL的多模遥感数据融合进行系统综述。更具体地说，首先给出了一些关于这个主题的基本知识。随后，进行了文献调查，分析了该领域的发展趋势。然后，从待融合数据模式的角度，回顾了多模式遥感数据融合中的一些主要子领域，即空间光谱、时空、光探测和测距光学、合成孔径雷达光学和遥感地理空间大数据融合。此外，我们还收集和总结了一些有价值的资源，以促进多模态遥感数据融合的发展。最后，强调了剩余的挑战和潜在的未来方向。

<br><br>

【2】 The Elements of Temporal Sentence Grounding in Videos: A Survey and Future Directions<br>
**标题**：视频中时间句的构成要素：综述与未来发展方向<br>
**链接**：https://arxiv.org/abs/2201.08071<br>

**作者**：Hao Zhang,Aixin Sun,Wei Jing,Joey Tianyi Zhou<br>
**机构**： and School of Computer Science and Engineering, Nanyang Technological University<br>
**摘要**：视频中的时态句子基础（TSGV），也称为自然语言视频定位（NLVL）或视频瞬间检索（VMR），旨在从未剪辑视频中检索语义上对应于语言查询的时态瞬间。TSGV连接了计算机视觉和自然语言，引起了这两个社区研究人员的极大关注。本调查试图对TSGV的基本概念、研究现状以及未来的研究方向进行总结。作为背景，我们以教程的形式介绍了TSGV中功能组件的通用结构：从原始视频的特征提取和语言查询，回答目标时刻的预测。然后，我们回顾了多模态理解和交互的技术，这是TSGV有效协调两种模态的关键焦点。我们构建了TSGV技术的分类体系，并详细阐述了不同类别的方法及其优缺点。最后，我们讨论了当前TSGV研究的问题，并分享了我们对未来研究方向的见解。

<br><br>

【3】 Multi-Modal Knowledge Graph Construction and Application: A Survey<br>
**标题**：多模态知识图构建与应用综述<br>
**链接**：https://arxiv.org/abs/2202.05786<br>

**作者**：Xiangru Zhu,Zhixu Li,Xiaodan Wang,Xueyao Jiang,Penglei Sun,Xuwu Wang,Yanghua Xiao,Nicholas Jing Yuan<br>
**机构**： Xiao are withthe School of Computer Science, Fudan University<br>
**摘要**：近年来，以知识图的快速增长为特征的知识工程再度兴起。然而，现有的大多数知识图都是用纯符号表示的，这损害了机器理解现实世界的能力。知识图的多模态化是实现人的机器智能的关键步骤。这一努力的结果是多模态知识图（MMKG）。在这篇关于由文本和图像构建的MMKG的综述中，我们首先给出了MMKG的定义，然后介绍了多模态任务和技术。然后，我们分别系统地回顾了MMKG构建和应用方面的挑战、进展和机遇，详细分析了不同解决方案的优缺点。我们用与MMKG相关的开放性研究问题来完成这项调查。

<br><br>

<a name="跟踪"/>

# 跟踪

【1】 Recent Advances in Embedding Methods for Multi-Object Tracking: A Survey<br>
**标题**：嵌入方法在多目标跟踪中的研究进展<br>
**链接**：https://arxiv.org/abs/2205.10766<br>

**作者**：Gaoang Wang,Mingli Song,Jenq-Neng Hwang<br>
**机构**：Zhejiang University<br>
**摘要**：多目标跟踪（MOT）旨在跨视频帧关联目标对象，以获得整个运动轨迹。随着深度神经网络的发展和对智能视频分析需求的不断增加，MOT在计算机视觉领域的兴趣显著增加。在MOT中，嵌入方法在目标位置估计和时间身份关联中起着至关重要的作用。与其他计算机视觉任务（如图像分类、目标检测、再识别和分割）不同，MOT中的嵌入方法有很大的变化，并且从未得到系统的分析和总结。本文首先从补丁级嵌入、单帧嵌入、跨帧联合嵌入、相关嵌入、序列嵌入、tracklet嵌入和跨轨关系嵌入七个不同的角度对MOT中的嵌入方法进行了全面的概述和深入的分析。我们进一步总结了现有广泛使用的MOT数据集，并根据嵌入策略分析了现有最先进方法的优势。最后，讨论了一些尚待研究的关键领域和未来的研究方向。

<br><br>

【2】 Single Object Tracking Research: A Survey<br>
**标题**：单目标跟踪研究综述<br>
**链接**：https://arxiv.org/abs/2204.11410<br>

**作者**：Ruize Han,Wei Feng,Qing Guo,Qinghua Hu<br>
**机构**： Tianjin University<br>
**摘要**：视觉目标跟踪是计算机视觉中的一项重要任务，在视频监控、视觉导航等领域有着广泛的应用。视觉目标跟踪也有很多挑战，例如目标遮挡和变形。为了解决上述问题，准确有效地跟踪目标，近年来出现了许多跟踪算法。本文介绍了近十年来两种最流行的跟踪框架的原理和代表性工作，即用于目标跟踪的相关滤波器和暹罗网络。然后根据网络结构的不同，提出了一些基于深度学习的跟踪方法。我们还介绍了一些经典的策略来应对跟踪问题中的挑战。此外，本文详细介绍和比较了视觉跟踪的基准和挑战，总结了视觉跟踪的发展历史和发展趋势。重点介绍了目标跟踪的未来发展，我们认为在一些需要解决的问题之前，目标跟踪将应用于现实场景中，如长期跟踪、低功耗高速跟踪和攻击鲁棒跟踪等问题。未来，深度图像、热图像和传统彩色图像等多模态数据的集成将为视觉跟踪提供更多解决方案。此外，跟踪任务将与其他一些任务一起进行，例如视频对象检测和分割。

<br><br>

【3】 Optical tracking in team sports<br>
**标题**：团队运动中的光学跟踪<br>
**链接**：https://arxiv.org/abs/2204.04143<br>

**作者**：Pegah Rahimian,Laszlo Toka<br>
**机构**：Budapest University of Technology, and Economics, Budapest, Hungary, MTA-BME Information Systems, Research Group<br>
**摘要**：对教练、球探和球迷来说，体育分析变得至关重要。最近，计算机视觉研究人员提出了几种自动跟踪球员和球的方法，从而承担了收集必要数据的挑战。对球员进行量化分析，并对球员的表现进行跟踪。通过这项调查，我们的目标是为定量数据分析师提供关于创建输入数据的过程及其特征的基本理解。因此，我们总结了光学跟踪的最新方法，分别对传统方法和深度学习方法进行了全面分类。此外，我们还讨论了跟踪的预处理步骤、该领域最常见的挑战，以及跟踪数据在运动队中的应用。最后，我们根据成本和局限性对这些方法进行了比较，并通过强调潜在的未来研究方向来总结工作。

<br><br>

【4】 A Survey for Deep RGBT Tracking<br>
**标题**：深度RGBT跟踪技术综述<br>
**链接**：https://arxiv.org/abs/2201.09296<br>

**作者**：Zhangyong Tang,Tianyang Xu,Xiao-Jun Wu<br>
**机构**： Wu are with the School of Artificial Intelligenceand Computer Science, Jiangnan University<br>
**摘要**：利用可见光（RGB）和热红外（TIR）电磁波进行视觉目标跟踪，简称RGBT跟踪，近年来在跟踪领域受到越来越多的关注。鉴于深度学习技术的迅速发展，本文对基于深度神经网络的RGBT跟踪器进行了综述。首先，我们简要介绍了归为这一类的RGBT跟踪器。然后，对现有的RGBT跟踪器在几个具有挑战性的基准上进行了统计比较。具体来说，MDNet和暹罗体系结构是RGBT社区的两个主流框架，尤其是前者。基于MDNet的跟踪器实现了更高的性能，而基于暹罗的跟踪器满足了实时性要求。总之，由于发布了大规模数据集LasHeR，应进一步考虑端到端框架的集成，例如暹罗和Transformer，以实现实时性和更稳健的性能。此外，在设计网络时应更多地考虑数学意义。对于关注RGBT追踪的研究人员来说，这项调查可以被视为一个查找表。

<br><br>

【5】 Single Object Tracking: A Survey of Methods, Datasets, and Evaluation Metrics<br>
**标题**：单目标跟踪：方法、数据集和评估度量综述<br>
**链接**：https://arxiv.org/abs/2201.13066<br>

**作者**：Zahra Soleimanitaleb,Mohammad Ali Keyvanrad<br>
**摘要**：目标跟踪是计算机视觉中最重要的任务之一，有许多常识性的应用，如交通监控、机器人、自动车辆跟踪等。之后很长一段时间，人们尝试了不同的研究，但由于各种各样的挑战，例如遮挡、光照变化、快速运动等，该领域的研究仍在继续。本文考察了以下对象的不同策略，并给出了一个综合分类，将以下策略分为四个基本类别：基于特征的、基于分割的、基于估计的和基于学习的方法，每个方法都有其索赔子类别。本文主要研究基于学习的策略，将其分为生成性策略、辨别性策略和强化学习三类。这场歧视性节目的一个子类是深度学习。由于高性能，深学习一直以来都是非常多的考虑。最后，将介绍最常用的不同数据集和评估方法。


<br><br>

<a name="超分辨率去噪去模糊去雾"/>

# **超分辨率|去噪|去模糊|去雾**

【1】 Single Image Super-Resolution Methods: A Survey<br>
**标题**：单幅图像超分辨率方法综述<br>
**链接**：https://arxiv.org/abs/2202.11763<br>

**作者**：Bahattin Can Maral<br>
**摘要**：超分辨率（SR）是从同一场景的一个或多个低分辨率观测值中获取高分辨率图像的过程，在过去几十年中，它一直是信号处理和图像处理领域的一个非常热门的研究课题。由于最近卷积神经网络的发展，随着进入门槛的显著降低，SR算法的普及率急剧上升。最近，这种流行已经蔓延到视频处理领域，发展出了实时工作的SR模型。在这篇论文中，我们比较了专门从事单一图像处理的不同SR模型，并将简要介绍它们是如何在过去的几年中演变成具有许多不同目标和形状的。

<br><br>

【2】 Deep Image Deblurring: A Survey<br>
**标题**：深度图像去模糊：综述<br>
**链接**：https://arxiv.org/abs/2201.10700<br>

**作者**：Kaihao Zhang,Wenqi Ren,Wenhan Luo,Wei-Sheng Lai,Bjorn Stenger,Ming-Hsuan Yang,Hongdong Li<br>
**摘要**：图像去模糊是低级计算机视觉中的一个经典问题，其目的是从模糊的输入图像中恢复清晰的图像。深度学习的最新进展已经在解决这个问题上取得了重大进展，并且已经提出了大量的去模糊网络。本文对最近发表的基于深度学习的图像去模糊方法进行了全面而及时的调查，旨在为社区提供有用的文献综述。我们首先讨论图像模糊的常见原因，介绍基准数据集和性能指标，并总结不同的问题公式。接下来，我们将根据体系结构、损失函数和应用，对使用卷积神经网络（CNN）的方法进行分类，并提供详细的回顾和比较。此外，我们还讨论了一些特定领域的去模糊应用，包括人脸图像、文本和立体图像对。最后，我们讨论了主要挑战和未来的研究方向。

<br><br>

【3】 A Survey on Image Deblurring<br>
**标题**：图像去模糊技术综述<br>
**链接**：https://arxiv.org/abs/2202.07456<br>

**作者**：ChuMiao Li<br>
**机构**：Xihua University<br>
**摘要**：随着社会生活质量的提高和日常工作的实际需要，图像越来越多地出现在我们身边。由于相机抖动、人体运动等原因造成的图像模糊已成为影响图像质量的关键。如何去除图像模糊，恢复清晰的图像已逐渐成为计算机视觉领域的一个重要研究方向。经过半个多世纪的不懈努力，广大科技工作者在图像去模糊方面取得了丰硕的成果。本文回顾了图像去模糊的工作，并具体介绍了更多经典的图像去模糊方法，这有助于了解当前的研究和展望未来的发展趋势。本文回顾了传统的图像去模糊方法和深度表示的图像去模糊方法，并对相应的技术方法进行了全面的分类和介绍。本文的综述可以为图像去模糊领域的研究者提供一些指导，同时也有利于他们以后的研究和研究。

<br><br>

【4】 Optical Flow for Video Super-Resolution: A Survey<br>
**标题**：视频超分辨率光流研究综述<br>
**链接**：https://arxiv.org/abs/2203.10462<br>

**作者**：Zhigang Tu,Hongyan Li,Wei Xie,Yuanzhong Liu,Shifu Zhang,Baoxin Li,Junsong Yuan<br>
**摘要**：视频超分辨率是当前计算机视觉领域最活跃的研究课题之一，它在许多视觉应用中发挥着重要作用。通常，视频超分辨率包含一个重要组件，即运动补偿，用于估计连续视频帧之间的位移，以便进行时间对齐。光流可以在连续帧之间提供密集的亚像素运动，是完成这项任务最常用的方法之一。为了更好地理解光流在视频超分辨率中的作用，在这项工作中，我们首次对这一主题进行了全面的综述。本研究涉及以下主要主题：超分辨率的功能（即为什么我们需要超分辨率）；视频超分辨率的概念（即什么是视频超分辨率）；评估指标的描述（即（视频）超分辨率的表现）；介绍了基于光流的视频超分辨率技术；利用光流捕捉视频超分辨率的时间相关性的研究。重点对基于深度学习的视频超分辨率方法进行了深入研究，并对一些有代表性的算法进行了分析和比较。此外，我们还强调了一些有希望的研究方向和有待进一步解决的开放性问题。

<br><br>

【5】 A Survey of Super-Resolution in Iris Biometrics with Evaluation of Dictionary-Learning<br>
**标题**：基于词典学习评价的虹膜生物识别超分辨率研究<br>
**链接**：https://arxiv.org/abs/2203.14203<br>

**作者**：F. Alonso-Fernandez,R. A. Farrugia,J. Bigun,J. Fierrez,E. Gonzalez-Sosa<br>
**机构**：se) 2Department of Communications and Computer Engineering (CCE), University of Malta<br>
**摘要**：分辨率的缺乏对基于图像的生物特征识别的性能有负面影响。虽然已经提出了许多通用的超分辨率方法来恢复低分辨率图像，但它们通常旨在增强图像的视觉外观。然而，生物特征图像的视觉增强并不一定与更好的识别性能相关。因此，重建方法需要结合来自目标生物特征模式的特定信息，以有效提高识别率。本文对文献中提出的虹膜超分辨率方法进行了综述。我们还采用了基于局部图像块PCA特征变换的特征块重建方法。虹膜的结构是通过构建一个依赖于补丁位置的字典来利用的。此外，图像面片被单独恢复，具有自己的重建权重。这允许对解决方案进行局部优化，有助于保存局部信息。为了评估该算法，我们对CASIA Interval V3数据库中的高分辨率图像进行了降级。考虑了不同的恢复，分辨率为15x15像素。据我们所知，这是文献中使用的最小分辨率之一。该框架由六个公共虹膜比较器进行补充，用于进行生物特征验证和识别实验。实验结果表明，在很低的分辨率下，该方法明显优于双线性插值和双三次插值。当考虑到只有15x15像素的虹膜图像时，许多比较器的性能达到了令人印象深刻的等错误率，低至5%，最高精度为77-84%。这些结果清楚地证明了在匹配之前使用经过训练的超分辨率技术来提高虹膜图像质量的好处。

<br><br>

<a name="人脸"/>

# 人脸

【1】 A comprehensive survey on semantic facial attribute editing using generative adversarial networks<br>
**标题**：基于产生式对抗网络的人脸属性语义编辑研究综述<br>
**链接**：https://arxiv.org/abs/2205.10587<br>

**作者**：Ahmad Nickabadi,Maryam Saeedi Fard,Nastaran Moradzadeh Farid,Najmeh Mohammadbagheri<br>
**机构**：Computer Engineering Department, Amirkabir University of Technology, Tehran, Iran<br>
**摘要**：由于深度卷积神经网络和生成模型的发展，随机照片真实感图像的生成在过去几年中经历了巨大的增长。在不同的领域中，人脸照片受到了广泛的关注，并提出了大量的人脸生成和操作模型。语义面部属性编辑是在不影响图像的其他属性的情况下，改变面部图像的一个或多个属性值的过程。请求的修改以属性向量或驱动人脸图像的形式提供，整个过程由相应的模型执行。本文综述了语义人脸属性编辑的最新研究成果和进展。我们涵盖了这些模型的所有相关方面，包括相关定义和概念、体系结构、损失函数、数据集、评估指标和应用程序。根据它们的体系结构，将最先进的模型分类并研究为编码器-解码器、图像到图像和光引导模型。还讨论了当前最先进方法的挑战和限制。

<br><br>

【2】 Video-based Facial Micro-Expression Analysis: A Survey of Datasets, Features and Algorithms<br>
**标题**：基于视频的人脸微表情分析：数据集、特征和算法综述<br>
**链接**：https://arxiv.org/abs/2201.12728<br>

**作者**：Xianye Ben,Yi Ren,Junping Zhang,Su-Jing Wang,Kidiyo Kpalma,Weixiao Meng,Yong-Jin Liu<br>
**摘要**：与传统的面部表情不同，微表情是不自觉的、短暂的面部表情，能够揭示人们试图隐藏的真实情感。因此，它们可以在广泛的应用中提供重要信息，如测谎、刑事检测等。由于微表情是瞬时的，强度较低，但是，它们的检测和识别非常困难，并且严重依赖专家经验。由于其固有的特殊性和复杂性，基于视频的微表情分析具有很大的吸引力和挑战性，近年来已成为一个活跃的研究领域。尽管在这一领域有许多进展，但迄今为止，还没有一项全面的调查为研究人员提供对这些进展的系统概述和统一的评估。因此，在这篇调查论文中，我们首先强调宏观和微观表达之间的关键差异，然后利用这些差异指导我们对级联结构中基于视频的微观表达分析的研究调查，包括神经心理学基础、数据集、特征、定位算法、识别算法、，最先进方法的应用和评估。对于每一个方面，基本技术、先进发展和主要挑战都会得到解决和讨论。此外，在考虑了现有微表情数据集的局限性后，我们提出并发布了一个新的数据集，称为微表情和宏表情仓库（MMEW），其中包含更多视频样本和更多标记的情感类型。然后，我们分别在CAS（ME）2和MMEW及SAMM上对用于识别的代表性方法进行了统一比较。最后，展望了未来的研究方向。

<br><br>

<a name="视觉解释视频理解VQAcaption等"/>

# **视觉解释|视频理解VQA|caption等**

【1】 Deep Learning Approaches on Image Captioning: A Review<br>
**标题**：图像字幕深度学习方法综述<br>
**链接**：https://arxiv.org/abs/2201.12944<br>

**作者**：Taraneh Ghandi,Hamidreza Pourreza,Hamidreza Mahyar<br>
**摘要**：自动图像字幕是一个具有挑战性的问题，涉及到对图像内容的描述，在各个研究领域有着广泛的应用。一个显著的例子是为视力受损者设计助手。近年来，由于深度学习的突破，图像字幕方法取得了重大进展。这篇调查论文旨在对最新的图像字幕技术及其性能进行结构化回顾，主要关注深度学习方法。我们还回顾了广泛使用的数据集和性能指标，以及关于图像字幕中存在的问题和未解决的挑战的讨论。

<br><br>

<a name="时序行为识别姿态视频运动估计"/>

# **时序|行为识别|姿态|视频|运动估计**

【1】 A Survey of Video-based Action Quality Assessment<br>
**标题**：基于视频的动作质量评估研究综述<br>
**链接**：https://arxiv.org/abs/2204.09271<br>

**作者**：Shunli Wang,Dingkang Yang,Peng Zhai,Qing Yu,Tao Suo,Zhan Sun,Ka Li,Lihua Zhang<br>
**机构**：Institute of AI & Robotics, Fudan University , Shanghai, China, ZhongShan Hospital , Shanghai, China, Ji Hua Laboratory , Foshan, China<br>
**摘要**：人体行为识别与分析在视频监控、视频检索、人机交互等领域有着巨大的需求和重要的应用意义。人类行为质量评估的任务要求智能系统自动、客观地评估人类完成的行为。行动质量评估模型可以减少行动评估所花费的人力和物力，减少主观能动性。在本文中，我们提供了一个全面的调查现有的论文视频为基础的行动质量评估。与人类行为识别不同，行为质量评估的应用场景相对狭窄。现有的大部分工作侧重于体育和医疗保健。我们首先介绍了人类行为质量评估的定义和挑战。然后，我们介绍了现有的数据集和评估指标。此外，我们还根据模式分类总结了体育和医疗保健的方法，并根据这两个领域的特点总结了出版机构。最后，结合最近的工作，讨论了行动质量评估的发展方向。

<br><br>

【2】 2D Human Pose Estimation: A Survey<br>
**标题**：二维人体姿态估计研究综述<br>
**链接**：https://arxiv.org/abs/2204.07370<br>

**作者**：Haoming Chen,Runyang Feng,Sifan Wu,Hao Xu,Fengcheng Zhou,Zhenguang Liu<br>
**摘要**：人体姿势估计旨在定位输入数据（例如图像、视频或信号）中的人体解剖关键点或身体部位。它是使机器能够深刻理解人类行为的关键组成部分，并已成为计算机视觉和相关领域的一个突出问题。深度学习技术允许直接从数据中学习特征表示，极大地提高了人体姿势估计的性能边界。在本文中，我们总结了二维人体姿势估计方法的最新成果，并对其进行了全面综述。简单地说，现有的方法将他们的努力放在三个方向上，即网络架构设计、网络训练细化和后处理。网络架构设计着眼于人体姿势估计模型的架构，为关键点识别和定位提取更稳健的特征。网络训练精化利用神经网络的训练，旨在提高模型的表示能力。后处理进一步结合了模型无关的抛光策略，以提高关键点检测的性能。这项调查涉及200多项研究贡献，包括方法框架、通用基准数据集、评估指标和绩效比较。我们试图为研究人员提供一个更全面、更系统的人体姿势估计综述，使他们能够获得一个宏大的全景，更好地确定未来的方向。

<br><br>

【3】 A Survey on Infrared Image and Video Sets<br>
**标题**：红外图像和视频设备综述<br>
**链接**：https://arxiv.org/abs/2203.08581<br>

**作者**：Kevser Irem Danaci,Erdem Akagunduz<br>
**机构**：Department of Electrical Engineering, Sivas University of Science And Technology, Turkey., The Graduate School of Informatics, Middle East Technical University, Turkey.<br>
**摘要**：在这项调查中，我们为人工智能和计算机视觉研究人员编制了一份公开的红外图像和视频集列表。我们主要关注红外图像和视频集，这些图像和视频集是为计算机视觉应用而收集和标记的，如目标检测、目标分割、分类和运动检测。我们根据传感器类型、图像分辨率和比例对92种不同的公共可用或私人设备进行分类。我们详细描述了每一套装置的收集目的、操作环境、光学系统特性和应用领域。我们还概述了与红外图像相关的基本概念，如红外辐射、红外探测器、红外光学和应用领域。我们从不同的角度分析了整个语料库的统计意义。我们相信，这项调查将为计算机视觉和人工智能研究人员提供指导，这些研究人员对研究可见光范围以外的光谱感兴趣。

<br><br>

【4】 3D Human Motion Prediction: A Survey<br>
**标题**：三维人体运动预测研究综述<br>
**链接**：https://arxiv.org/abs/2203.01593<br>

**作者**：Kedi Lyu,Haipeng Chen,Zhenguang Liu,Beiqi Zhang,Ruili Wang<br>
**机构**：Jilin University, Changchun, Jilin, China, Zhejiang Gongshang University, Hangzhou, Zhejiang, China, Sichuan University, Chengdu, Sichuan, China, A R T I C L E I N F O<br>
**摘要**：三维人体运动预测（3D human motion prediction，3D human motion prediction）是计算机视觉和机器智能领域中一个具有重要意义和挑战性的问题，它可以帮助机器理解人类行为。随着对深度神经网络（DNN）的不断发展和理解，以及大规模人体运动数据集的可用性，人体运动预测在学术界和工业界引起了极大的兴趣。在此背景下，对三维人体运动预测进行了全面的调查，目的是回顾和分析现有文献中的相关工作。此外，还构建了一个相关的分类法，对现有的3D人体运动预测方法进行分类。在本次调查中，相关方法分为三类：人体姿势表示、网络结构设计和预测目标。我们系统地回顾了自2015年以来人类运动预测领域的所有相关期刊和会议论文，这些论文根据本次调查中提出的分类进行了详细介绍。此外，本文还分别介绍了公共基准数据集、评估标准和性能比较。还讨论了最新方法的局限性，希望为未来的探索铺平道路。

<br><br>

【5】 A survey of top-down approaches for human pose estimation<br>
**标题**：自顶向下的人体姿态估计方法综述<br>
**链接**：https://arxiv.org/abs/2202.02656<br>

**作者**：Thong Duy Nguyen,Milan Kresovic<br>
**摘要**：二维图像视频中的人体姿势估计由于其在改善人类生活方面的巨大优势和潜在应用，例如行为识别、运动捕捉和增强现实、训练机器人和运动跟踪，近年来一直是计算机视觉领域的一个热门话题。许多通过深度学习实现的最先进方法已经解决了一些挑战，并在人体姿势估计领域带来了巨大的显著成果。方法分为两类：两步框架（自上而下方法）和基于零件的框架（自下而上方法）。虽然两步框架首先包含一个人物检测器，然后独立地估计每个盒子内的姿势，但检测图像中的所有身体部位以及关联属于不同人物的部位是在基于部位的框架中进行的。本文旨在为新手提供基于2D图像的深度学习方法的广泛回顾，用于识别人的姿势，自2016年以来只关注自上而下的方法。通过本文的讨论，根据数学背景、挑战和局限性、基准数据集、评估指标以及方法之间的比较，提出了重要的检测器和估计器。

<br><br>

【6】 Video Question Answering: Datasets, Algorithms and Challenges<br>
**标题**：视频问答：数据集、算法和挑战<br>
**链接**：https://arxiv.org/abs/2203.01225<br>

**作者**：Yaoyao Zhong,Wei Ji,Junbin Xiao,Yicong Li,Weihong Deng,Tat-Seng Chua<br>
**机构**：National University of Singapore, Beijing University of Posts and Telecommunications<br>
**摘要**：视频问答（VideoQA）旨在根据给定的视频回答自然语言问题。随着联合视觉和语言理解的最新研究趋势，它越来越受到关注。然而，与ImageQA相比，VideoQA在很大程度上没有得到充分的探索，进展缓慢。尽管不同的算法不断被提出，并在不同的VideoQA数据集上取得了成功，但我们发现，缺乏有意义的调查来对它们进行分类，这严重阻碍了其发展。因此，本文对VideoQA进行了清晰的分类和全面的分析，重点介绍了数据集、算法和独特的挑战。然后，我们指出了从因素质量保证到推理质量保证对视频内容认知的研究趋势，最后，我们总结了一些有希望的未来探索方向。

<br><br>

【7】 Recovering 3D Human Mesh from Monocular Images: A Survey<br>
**标题**：从单目图像中恢复三维人体网格的研究进展<br>
**链接**：https://arxiv.org/abs/2203.01923<br>

**作者**：Yating Tian,Hongwen Zhang,Yebin Liu,Limin Wang<br>
**机构**： It is also involved in widespread appli-•Yating Tian and Limin Wang are with the State Key Laboratory forNovel Software Technology at Nanjing University, cn•HongwenZhangandYebinLiuarewiththeDepartmentofAutomationatTsinghuaUniversity<br>
**摘要**：从单目图像估计人体姿势和形状是计算机视觉中一个长期存在的问题。自统计人体模型发布以来，三维人体网格恢复一直受到广泛关注。为了获得对齐良好且物理上合理的网格结果，开发了两种范式，以克服二维到三维提升过程中的挑战：i）基于优化的范式，其中利用不同的数据项和正则化项作为优化目标；ii）基于回归的范式，采用深度学习技术以端到端的方式解决问题。同时，持续的努力致力于提高各种数据集的3D网格标签的质量。尽管在过去十年中取得了显著的进展，但由于身体运动灵活、外观多样、环境复杂，以及野外注释不足，这项任务仍然具有挑战性。据我们所知，这是第一次调查，重点是单眼三维人体网格恢复的任务。我们首先介绍身体模型，然后通过深入分析它们的优缺点，介绍恢复框架和训练目标。我们还总结了数据集、评估指标和基准测试结果。最后讨论了有待解决的问题和未来的发展方向，希望能激发研究人员的积极性，促进他们在这一领域的研究。定期更新的项目页面可在https://github.com/tinatiansjz/hmr-survey.

<br><br>

【8】 Didn't see that coming: a survey on non-verbal social human behavior forecasting<br>
**标题**：没有预见到这一点：一项关于非语言社会人类行为预测的调查<br>
**链接**：https://arxiv.org/abs/2203.02480<br>

**作者**：German Barquero,Johnny Núñez,Sergio Escalera,Zhen Xu,Wei-Wei Tu,Isabelle Guyon,Cristina Palmero<br>
**机构**：Johnny N´u˜nez, Universitat de Barcelona and Computer Vision Center, Spain, Paradigm, Beijing, China, LISN (CNRSINRIA) Universit´e Paris-Saclay, France, and ChaLearn, USA<br>
**摘要**：近年来，非言语社会人类行为预测越来越引起研究界的兴趣。它在人机交互和社会感知人类运动生成方面的直接应用使其成为一个非常有吸引力的领域。在这项调查中，我们以一种通用的方式定义了多个交互代理的行为预测问题，旨在统一传统上分离的社会信号预测和人体运动预测领域。我们认为，两种问题表述都涉及同一个概念问题，并确定了许多共同的基本挑战：未来随机性、上下文意识、历史利用，等等。我们还提出了一个分类法，其中包括过去5年中以非常翔实的方式发布的方法，并描述了当前社区对这个问题的主要关注。为了促进这一领域的进一步研究，我们还提供了非行为社交互动视听数据集的概述。最后，我们描述了这项任务中使用的最常见的指标及其特殊问题。

<br><br>

【9】 Spatiotemporal Data Mining: A Survey<br>
**标题**：时空数据挖掘：综述<br>
**链接**：https://arxiv.org/abs/2206.12753<br>

**作者**：Arun Sharma,Zhe Jiang,Shashi Shekhar<br>
**摘要**：时空数据挖掘旨在发现大空间和时空数据中有趣、有用但不平凡的模式。它们被用于各种应用领域，如公共安全、生态学、流行病学、地球科学等。由于虚假模式的高社会成本和过高的计算成本，这一问题具有挑战性。由于快速增长，最近的时空数据挖掘调查需要更新。此外，他们没有充分调查时空数据挖掘的并行技术。本文提供了时空数据挖掘方法的最新综述。此外，它还详细介绍了时空数据挖掘的并行公式。

<br><br>

【10】 Efficient Annotation and Learning for 3D Hand Pose Estimation: A Survey<br>
**标题**：三维手势估计的高效标注和学习：综述<br>
**链接**：https://arxiv.org/abs/2206.02257<br>

**作者**：Takehiko Ohkawa,Ryosuke Furuta,Yoichi Sato<br>
**机构**：Institute of Industrial Science, The University of Tokyo,-,-, Komaba, Meguro-ku, Tokyo,-, Japan.<br>
**摘要**：在这篇综述中，我们从有效注释和学习的角度对三维手姿势估计进行了全面的分析。特别是，我们研究了三维手姿势标注的最新方法和有限标注数据的学习方法。在三维手姿势估计中，收集三维手姿势注释是开发手姿势估计器及其应用（如视频理解、AR/VR和机器人）的关键步骤。然而，获取带注释的三维手姿势很麻烦，例如，由于难以访问三维信息和遮挡。为了阐明最近的工作如何解决注释问题，我们研究了分为手动、基于合成模型、基于手传感器和计算方法的注释方法。由于这些标注方法并不总是在大规模上可用，我们研究了在没有足够标注数据的情况下学习三维手姿势的方法，即自我监督预训练、半监督学习和领域自适应。基于对这些有效注释和学习的分析，我们进一步讨论了该领域的局限性和未来可能的方向。


<br><br>

<a name="自动驾驶车辆车道检测"/>

# **自动驾驶|车辆|车道检测**

【1】 Surround-view Fisheye Camera Perception for Automated Driving: Overview, Survey and Challenges<br>
**标题**：用于自动驾驶的环视鱼眼相机感知：概述、综述和挑战<br>
**链接**：https://arxiv.org/abs/2205.13281<br>

**作者**：Varun Ravi Kumar,Ciaran Eising,Christian Witt,Senthil Yogamani<br>
**机构**： of Electronic and Computer Engineering at theUniversity of Limerick<br>
**摘要**：环绕视图鱼眼摄像头通常用于自动驾驶中的近场传感。车辆四个侧面上的四个鱼眼摄像头足以覆盖车辆周围360度的区域，捕捉整个近场区域。一些主要用例是自动停车、交通堵塞辅助和城市驾驶。由于汽车感知的主要焦点是远场感知，因此数据集有限，近场感知任务的研究很少。与远场相比，由于10cm的高精度目标检测要求和目标的局部可见性，环绕视图感知带来了额外的挑战。由于鱼眼摄像头的径向畸变较大，标准算法无法轻松扩展到环绕视图用例。因此，我们有动机为研究人员和从业者提供一个独立的汽车鱼眼摄像头感知参考。首先，我们对常用的鱼眼相机模型进行了统一和分类处理。其次，我们讨论了各种感知任务和现有文献。最后，我们讨论了挑战和未来方向。

<br><br>

【2】 A Review on Viewpoints and Path-planning for UAV-based 3D Reconstruction<br>
**标题**：无人机三维重建视点与路径规划研究综述<br>
**链接**：https://arxiv.org/abs/2205.03716<br>

**作者**：Mehdi Maboudi,MohammadReza Homaei,Soohwan Song,Shirin Malihi,Mohammad Saadatseresht,Markus Gerke<br>
**机构**：Institute of Geodesy and Photogrammetry, Technische Universität Braunschweig, Germany, School of Surveying and Geospatial Eng., College of Eng., University of Tehran, Tehran , Iran, Intelligent Robotics Research Division, ETRI, Daejeon , Republic of Korea<br>
**摘要**：无人机（UAV）是广泛使用的平台，用于承载各种应用的数据捕获传感器。这种成功的原因可以在很多方面找到：无人机的高机动性、执行自动数据采集的能力、在不同高度飞行的能力，以及到达几乎任何有利位置的可能性。选择合适的视点并规划无人机的最佳轨迹是一个新兴主题，旨在提高数据采集过程的自动化、效率和可靠性，以获得具有所需质量的数据集。另一方面，利用无人机获取的数据进行三维重建也吸引了研究和工业界的关注。本文综述了用于大规模物体三维重建的视点和路径规划的各种无模型和基于模型的算法。所分析的方法仅限于使用单个无人机作为数据捕获平台进行户外三维重建的方法。除了讨论评估策略外，本文还强调了研究方法的创新和局限性。最后对现有的挑战和未来的研究前景进行了批判性分析。

<br><br>

【3】 Multi-modal Sensor Fusion for Auto Driving Perception: A Survey<br>
**标题**：多模态传感器融合在汽车驾驶感知中的研究进展<br>
**链接**：https://arxiv.org/abs/2202.02703<br>

**作者**：Keli Huang,Botian Shi,Xiang Li,Xin Li,Siyuan Huang,Yikang Li<br>
**摘要**：多模态融合是自动驾驶系统感知的一项基本任务，最近引起了许多研究人员的兴趣。然而，由于原始数据噪声大、信息利用率低以及多模态传感器的失调，实现相当好的性能并非易事。在本文中，我们对现有的基于多模态的自主驾驶感知任务方法进行了文献综述。一般来说，我们做了详细的分析，包括超过50篇论文，利用感知传感器，包括激光雷达和相机，试图解决目标检测和语义分割任务。与传统的融合模型分类方法不同，我们提出了一种创新的方法，从融合阶段的角度，通过更合理的分类法将融合模型分为两大类、四小类。此外，我们深入研究了当前的融合方法，关注剩余的问题，并就潜在的研究机会展开讨论。总之，我们希望在本文中为自主驾驶感知任务提出一种新的多模态融合方法分类，并引发对未来基于融合技术的思考。

<br><br>

<a name="图像分割"/>

# 图像分割

【1】 A State-of-the-art Survey of U-Net in Microscopic Image Analysis: from Simple Usage to Structure Mortification<br>
**标题**：U-net在显微图像分析中的研究现状：从简单使用到结构化屈辱<br>
**链接**：https://arxiv.org/abs/2202.06465<br>

**作者**：Jian Wu,Wanli Liu,Chen Li,Tao Jiang,Islam Mohammad Shariful,Hongzan Sun,Xiaoqi Li,Xintong Li,Xinyu Huang,Marcin Grzegorzek<br>
**摘要**：图像分析技术用于解决人工传统方法在疾病、废水处理、环境变化监测分析中的不足，卷积神经网络（CNN）在显微图像分析中发挥着重要作用。图像分割是检测、跟踪、监控、特征提取、建模和分析的一个重要步骤，其中U-Net在显微图像分割中的应用越来越多。本文全面回顾了U-Net的发展历史，分析了U-Net出现以来各种分割方法的各种研究成果，并对相关文献进行了综合评述。本文首先总结了U-Net的改进方法，然后列举了图像分割技术的现有意义以及这些年来的改进。最后，针对不同论文中U-Net的不同改进策略，按照详细的技术分类对各个应用目标的相关工作进行了回顾，以便于今后的研究。研究人员可以清楚地看到技术发展传播的动态，并跟上这一跨学科领域的未来趋势。

<br><br>

<a name="gan"/>

# gan

【1】 Generative Adversarial Networks for Image Super-Resolution: A Survey<br>
**标题**：图像超分辨率产生式对抗网络研究综述<br>
**链接**：https://arxiv.org/abs/2204.13620<br>

**作者**：Chunwei Tian,Xuanyu Zhang,Jerry Chun-Wen Lin,Wangmeng Zuo,Yanning Zhang<br>
**摘要**：单图像超分辨率（SISR）在图像处理领域发挥着重要作用。最新的生成性对抗网络（GANs）可以在小样本的低分辨率图像上取得优异的效果。然而，很少有文献对SISR中的不同GAN进行综述。本文从不同的角度对GANs进行了比较研究。我们首先来看一下GANs的发展。其次，我们在图像应用程序的大样本和小样本中展示了GAN的流行架构。然后，从有监督、半监督和无监督的角度分析了基于GANs的图像超分辨率优化方法和判别学习的动机、实现和差异。接下来，我们通过SISR中的定量和定性分析，在公共数据集上比较这些流行的GAN的性能。最后，我们强调了GANs的挑战和SISR的潜在研究点。

<br><br>

【2】 A Comprehensive Survey on Data-Efficient GANs in Image Generation<br>
**标题**：图像生成中的数据高效GANS综述<br>
**链接**：https://arxiv.org/abs/2204.08329<br>

**作者**：Ziqiang Li,Xintian Wu,Beihao Xia,Jing Zhang,Chaoyue Wang,Bin Li<br>
**机构**：University of Science and Technology of China, Zhejiang University, Huazhong University of Science and Technology, The University of Sydney, JD Explore Academy<br>
**摘要**：生成性对抗网络在图像合成方面取得了显著的成就。GAN的这些成功依赖于大规模数据集，需要太多的成本。在训练数据有限的情况下，如何稳定GANs的训练过程，生成逼真的图像已经引起了越来越多的关注。数据高效GANs（DE GANs）面临的挑战主要来自三个方面：（i）训练和目标分布之间的不匹配，（ii）鉴别器的过度拟合，以及（iii）潜在空间和数据空间之间的不平衡。虽然已经提出了许多增强和训练前策略来缓解这些问题，但缺乏一个系统的调查来总结DE GANs的特性、挑战和解决方案。在本文中，我们从分销优化的角度重新审视和定义了DE GANs。我们总结并分析了德甘的挑战。同时，我们提出了一种分类法，将现有的方法分为三类：数据选择、GANs优化和知识共享。最后，我们试图强调当前的问题和未来的方向。

<br><br>

【3】 A Survey of Cross-Modality Brain Image Synthesis<br>
**标题**：跨模态脑图像合成研究综述<br>
**链接**：https://arxiv.org/abs/2202.06997<br>

**作者**：Guoyang Xie,Jinbao Wang,Yawen Huang,Yefeng Zheng,Feng Zheng,Yaochu Jin<br>
**机构**：Jin,†, NICE Group, University of Surrey, VIP Lab, Southern University of Science and Technology, Jarvis Lab, Tencent, Bielefeld University<br>
**摘要**：完全对齐和配对的多模式神经成像数据的存在证明了其在脑疾病诊断中的有效性。然而，收集完整的对齐和配对数据是不切实际的，甚至是奢侈的，因为实际困难可能包括高成本、长时间采集、图像损坏和隐私问题。一个现实的解决方案是探索无监督学习或半监督学习来合成缺失的神经影像数据。在本文中，我们倾向于从不同的角度来研究多模态脑图像合成任务，包括监督水平、模态合成的范围以及基于合成的下游任务。特别是，我们深入分析了跨模态脑图像合成如何提高不同下游任务的性能。最后，我们评估了这些挑战，并为这个社区提供了几个开放的方向。所有资源均可在https://github.com/M-3LAB/awesome-multimodal-brain-image-systhesis

<br><br>

【4】 A survey on GANs for computer vision: Recent research, analysis and taxonomy<br>
**标题**：用于计算机视觉的GANs综述：最新研究、分析和分类<br>
**链接**：https://arxiv.org/abs/2203.11242<br>

**作者**：Guillermo Iglesias,Edgar Talavera,Alberto Díaz-Álvarez<br>
**机构**：Departamento de Sistemas Inform´aticos – Universidad Polit´ecnica de Madrid, ETSISI - Campus Sur, CAlan Turing, sn, Madrid, Spain<br>
**摘要**：在过去几年中，深度学习领域发生了几次革命，主要是以生成性对抗网络（GAN）的巨大影响为主题。GAN不仅在定义其模型时提供了独特的架构，还产生了对社会产生直接影响的难以置信的结果。由于GANs带来的重大改进和新的研究领域，社区不断推出新的研究，几乎不可能跟上时代的步伐。我们的调查旨在提供GANs的总体概述，展示最新的体系结构、损失函数的优化、验证指标以及最广泛认可的变体的应用领域。将评估模型架构的不同变体的效率，并展示最佳应用领域；作为该过程的重要组成部分，将分析评估GANs性能的不同指标和常用的损失函数。本次调查的最终目的是总结GANs的进化和性能，这些GANs具有更好的结果，可以指导该领域的未来研究人员。

<br><br>

【5】 Facke: a Survey on Generative Models for Face Swapping<br>
**标题**：FAKE：面孔交换生成模型综述<br>
**链接**：https://arxiv.org/abs/2206.11203<br>

**作者**：Wei Jiang,Wentao Dong<br>
**机构**：Shanghai Jiaotong University, Shanghai, China<br>
**摘要**：在这项工作中，我们研究了主流神经生成模型在人脸交换任务上的性能。我们在CVAE、CGAN、CVAE-GAN和条件扩散模型上进行了实验。现有的经过精细训练的模型已经成功地制造出肉眼无法分辨的假面（Facke），并实现了高目标指标。我们对它们进行了比较，并分析了它们的优缺点。此外，我们提出了一些有希望的技巧，尽管它们不适用于此任务。

<br><br>

【6】 Adversarial Patch Attacks and Defences in Vision-Based Tasks: A Survey<br>
**标题**：视觉任务中对抗性补丁攻击与防御的研究进展<br>
**链接**：https://arxiv.org/abs/2206.08304<br>

**作者**：Abhijith Sharma,Yijun Bian,Phil Munz,Apurva Narayan<br>
**备注**：A. Sharma and Y. Bian share equal contribution<br>
**摘要**：近年来，由于对人工智能模型的安全性和鲁棒性缺乏信任，深度学习模型尤其是安全关键系统中的对抗性攻击越来越受到重视。然而，更原始的对抗性攻击可能在物理上不可行，或者需要一些难以访问的资源，如训练数据，这促使补丁攻击的出现。在本次调查中，我们对现有的对抗性补丁攻击技术进行了全面概述，旨在帮助感兴趣的研究人员快速跟上这一领域的进展。我们还讨论了开发对抗补丁的检测和防御的现有技术，旨在帮助社区更好地了解这一领域及其在现实世界中的应用。

<br><br>

【7】 Recent Advances for Quantum Neural Networks in Generative Learning<br>
**标题**：量子神经网络在生成性学习中的研究进展<br>
**链接**：https://arxiv.org/abs/2206.03066<br>

**作者**：Jinkai Tian,Xiaoyu Sun,Yuxuan Du,Shanshan Zhao,Qing Liu,Kaining Zhang,Wei Yi,Wanrong Huang,Chaoyue Wang,Xingyao Wu,Min-Hsiu Hsieh,Tongliang Liu,Wenjing Yang,Dacheng Tao<br>
**机构**： NanyangTechnological University, •Wenjing Yang is with Institute for Quantum Information & State KeyLaboratory of High Performance Computing, National University of Defense Technology<br>
**摘要**：量子计算机是下一代设备，有望实现经典计算机无法实现的计算。实现这一目标的主要方法是通过量子机器学习，特别是量子生成学习。由于量子力学固有的概率性质，有理由假设量子生成学习模型（QGLM）可能会超过经典模型。因此，量子GLM正受到量子物理和计算机科学界越来越多的关注，其中提出了可以在具有潜在计算优势的短期量子机器上高效实现的各种QGLM。本文从机器学习的角度综述了QGLMs的最新进展。特别是，我们将这些QGLM解释为经典生成学习模型的量子扩展，包括量子电路出生机器、量子生成对抗网络、量子Boltzmann机器和量子自动编码器。在此背景下，我们探讨了它们之间的内在联系和根本区别。我们进一步总结了QGLMs在传统机器学习任务和量子物理中的潜在应用。最后，我们讨论了QGLMs面临的挑战和进一步的研究方向。

<br><br>

<a name="NAS模型搜索"/>

# **NAS模型搜索**

【1】 SuperNet in Neural Architecture Search: A Taxonomic Survey<br>
**标题**：神经结构搜索中的超网：分类学综述<br>
**链接**：https://arxiv.org/abs/2204.03916<br>

**作者**：Stephen Cha,Taehyeon Kim,Hayeon Lee,Se-Young Yun<br>
**机构**：OSI Lab, KAIST, MLAI Lab, KAIST<br>
**摘要**：深度神经网络（Deep Neural Network，DNN）在图像分类、目标检测和语义分割等广泛的视觉识别任务中取得了重大进展。卷积结构的发展带来了昂贵的计算成本，从而提高了性能。此外，网络设计已经成为一项艰巨的任务，这是劳动密集型的，需要高水平的领域知识。为了缓解这些问题，人们研究了各种神经结构搜索方法，这些方法可以自动搜索最佳结构，从而获得性能优于人类设计的模型的令人印象深刻的模型。本综述旨在概述该研究领域的现有工作，并特别关注supernet优化，该优化构建了一个神经网络，通过使用权重共享将所有体系结构组装为其子模型。我们的目标是通过将supernet优化分类，将其作为文献中常见挑战的解决方案提出：数据端优化、排名相关性差缓解，以及针对许多部署场景的可转移NAS。

<br><br>

【2】 Neural Architecture Search for Dense Prediction Tasks in Computer Vision<br>
**标题**：计算机视觉中密集预测任务的神经结构搜索<br>
**链接**：https://arxiv.org/abs/2202.07242<br>

**作者**：Thomas Elsken,Arber Zela,Jan Hendrik Metzen,Benedikt Staffler,Thomas Brox,Abhinav Valada,Frank Hutter<br>
**机构**： 2 University of Freiburg<br>
**摘要**：近年来，深度学习的成功导致对神经网络体系结构工程的需求不断增长。因此，旨在以数据驱动方式而非手动方式自动设计神经网络架构的神经架构搜索（neural architecture search，NAS）已发展成为一个热门研究领域。随着跨体系结构权重共享策略的出现，NAS已适用于更广泛的问题。特别是，现在有许多出版物针对计算机视觉中需要像素级预测的密集预测任务，例如语义分割或对象检测。这些任务带来了新的挑战，例如高分辨率数据带来的更高内存占用、学习多尺度表示、更长的训练时间以及更复杂和更大的神经结构。在这篇手稿中，我们通过详细阐述这些新的挑战和调查解决这些挑战的方法，为密集预测任务提供了NAS概述，以便于未来研究和现有方法对新问题的应用。


<br><br>

<a name="表征学习"/>

# 表征学习

【1】 Empirical Evaluation and Theoretical Analysis for Representation Learning: A Survey<br>
**标题**：表征学习的实证评价与理论分析：综述<br>
**链接**：https://arxiv.org/abs/2204.08226<br>

**作者**：Kento Nozawa,Issei Sato<br>
**机构**：The University of Tokyo,RIKEN AIP<br>
**摘要**：表示学习使我们能够从数据集中自动提取通用特征表示，以解决另一个机器学习任务。最近，通过表示学习算法和简单预测器提取的特征表示在多个机器学习任务中表现出了最先进的性能。尽管表征学习取得了显著的进展，但由于表征学习的灵活性，根据应用的不同，存在各种评估表征学习算法的方法。为了了解当前的表征学习，我们回顾了表征学习算法的评估方法和理论分析。在我们的评估调查的基础上，我们还讨论了表征学习的未来方向。请注意，本调查是野泽和佐藤（2022）的扩展版。

<br><br>

【2】 Compositional Scene Representation Learning via Reconstruction: A Survey<br>
**标题**：基于重构的构图场景表征学习研究综述<br>
**链接**：https://arxiv.org/abs/2202.07135<br>

**作者**：Jinyang Yuan,Tonglin Chen,Bin Li,Xiangyang Xue<br>
**机构**：Shanghai Key Laboratory of Intelligent Information Processing, School of Computer Science, Fudan University<br>
**摘要**：视觉场景表征学习是计算机视觉领域的一个重要研究课题。如果对视觉场景学习更合适的表示，视觉任务的性能可以得到提高。复杂视觉场景是由相对简单的视觉概念组成，具有组合爆炸的特性。与直接表示整个视觉场景相比，提取合成场景表示可以更好地处理背景和对象的不同组合。由于合成场景表示抽象了对象的概念，因此基于这些表示执行视觉场景分析和理解可能更容易、更容易理解。此外，通过重建学习合成场景表示可以大大减少对训练数据注释的需求。因此，基于重构的合成场景表征学习具有重要的研究意义。在这篇综述中，我们首先讨论了在没有对象级监控的情况下从单个视点或多个视点学习的代表性方法，然后讨论了合成场景表示的应用，最后讨论了该主题的未来方向。

<br><br>

【3】 Vision-Language Intelligence: Tasks, Representation Learning, and Large Models<br>
**标题**：视觉-语言智能：任务、表征学习和大型模型<br>
**链接**：https://arxiv.org/abs/2203.01922<br>

**作者**：Feng Li,Hao Zhang,Yi-Fan Zhang,Shilong Liu,Jian Guo,Lionel M. Ni,PengChuan Zhang,Lei Zhang<br>
**机构**：International Digital Economy Academy (IDEA), Chinese Academy of Science ,Tsinghua University ,Microsoft Research, The Hong Kong University of Science and Technology (Guangzhou)<br>
**摘要**：本文从时间的角度对视觉语言智能进行了全面的综述。这项调查的灵感来自计算机视觉和自然语言处理方面的显著进展，以及最近从单模态处理转向多模态理解的趋势。我们将这一领域的发展总结为三个时期，即任务特定方法、视觉语言预训练（VLP）方法和由大规模弱标记数据授权的更大模型。我们首先以一些常见的VL任务为例，介绍任务特定方法的发展。然后，我们重点介绍了VLP方法，并全面回顾了模型结构和训练方法的关键组成部分。之后，我们将展示最近的工作如何利用大规模原始图像文本数据来学习与语言对齐的视觉表示，这些视觉表示在零镜头或少量镜头学习任务中表现得更好。最后，我们讨论了模态合作、统一表示和知识整合的一些潜在未来趋势。我们相信这篇综述将对人工智能和人工语言的研究人员和实践者有所帮助，尤其是那些对计算机视觉和自然语言处理感兴趣的人。

<br><br>

<a name="半弱无监督主动学习不确定性"/>

# **半弱无监督|主动学习|不确定性**

【1】 TOV: The Original Vision Model for Optical Remote Sensing Image Understanding via Self-supervised Learning<br>
**标题**：TOV：一种基于自监督学习的光学遥感图像理解视觉模型<br>
**链接**：https://arxiv.org/abs/2204.04716<br>

**作者**：Chao Tao,Ji Qia,Guo Zhang,Qing Zhu,Weipeng Lu,Haifeng Li<br>
**机构**：School of Geosciences and Info-Physics, Central South University, Changsha, Hunan, PR, State Key Laboratory of Information Engineering in Surveying, Mapping and Remote, Sensing, Wuhan University, Wuhan, Hubei, PR China, Southwest Jiaotong University<br>
**摘要**：通过有监督的数据依赖和任务依赖的方式训练模型，而不是以无标签和任务独立的方式训练人类视觉，我们是否走上了遥感图像理解（RSIU）的正确道路？我们认为，一个更理想的RSIU模型应该通过数据的内在结构而不是外在的人类标签来训练，以实现广泛RSIU任务的普遍性。根据这一假设，我们提出了遥感领域的原始视觉模型（TOV）。TOV模型通过大量未标记的光学数据，沿着从一般知识到专门知识的类似于人的自监督学习（SSL）路径进行训练，可以很容易地适应各种RSIU任务，包括场景分类、目标检测和语义分割，在12个公开的基准测试中，它的性能超过了主流的ImageNet监督预训练方法以及最近提出的两种SSL预训练方法。此外，我们还分析了两个关键因素对RSIU TOV模型构建性能的影响，包括使用不同数据采样方法的影响和自监督优化过程中学习路径的选择。我们相信，通过无标签和独立于任务的方式训练的通用模型可能是RSIU的下一个范例，并希望从本研究中提炼出的见解能够帮助开发RSIU的原始视觉模型。

<br><br>

【2】 Unsupervised Representation Learning for Point Clouds: A Survey<br>
**标题**：点云的无监督表示学习：综述<br>
**链接**：https://arxiv.org/abs/2202.13589<br>

**作者**：Aoran Xiao,Jiaxing Huang,Dayan Guan,Shijian Lu<br>
**摘要**：点云数据因其在各种不利情况下的高精度和鲁棒性而得到了广泛的研究。同时，深度神经网络（DNN）在监控和自动驾驶等各种应用中取得了令人印象深刻的成功。点云和DNN的融合产生了许多深层点云模型，这些模型在很大程度上是在大规模密集标记点云数据的监督下训练的。无监督点云表示学习（Unsupervised point cloud representation learning，简称Unsupervised point cloud representation learning，简称Unsupervised point cloud representation learning，简称Unsupervised learning，简称Unsupervised point cloud representation learning）旨在从未标记的点云数据。本文全面回顾了使用DNN的无监督点云表示学习。它首先描述了最近研究的动机、一般管道以及术语。然后简要介绍了相关背景，包括广泛采用的点云数据集和DNN体系结构。接下来，根据现有的无监督点云表示学习方法的技术方法，对其进行了广泛的讨论。我们还对多个广泛采用的点云数据集进行了定量基准测试和讨论。最后，我们就未来无监督点云表示学习研究中可能面临的一些挑战和问题发表了自己的拙见。与本次调查相关的项目已在https://github.com/xiaoaoran/3d_url_survey.

<br><br>

【3】 Survey on Self-supervised Representation Learning Using Image Transformations<br>
**标题**：基于图像变换的自监督表征学习综述<br>
**链接**：https://arxiv.org/abs/2202.08514<br>

**作者**：Muhammad Ali,Sayed Hashim<br>
**机构**：Mohamed Bin Zayed University of Artificial Intelligence, UAE<br>
**摘要**：深层神经网络需要大量的训练数据，而在现实世界中，用于训练目的的数据非常稀缺。为了解决这些问题，使用了自监督学习（SSL）方法。使用几何变换的SSL（GT）是一种用于无监督表示学习的简单而强大的技术。尽管有多篇调查论文对SSL技术进行了综述，但没有一篇论文只关注那些使用几何变换的技术。此外，这些方法还没有在综述它们的论文中得到深入介绍。我们提出这项工作的动机是，几何变换已被证明是无监督表征学习中强有力的监督信号。此外，许多这样的作品获得了巨大的成功，但没有得到太多的关注。我们简要介绍了使用几何变换的SSL方法。我们列出了六个使用图像变换的代表性模型，包括基于预测和自动编码变换的模型。我们回顾了他们的体系结构和学习方法。我们还比较了这些模型在CIFAR-10和ImageNet数据集上的对象识别任务中的性能。我们的分析表明，AETv2在大多数情况下表现最好。在某些设置下，带有特征解耦的旋转也表现良好。然后，我们从观察结果中得出见解。最后，我们总结了研究结果和见解，强调了有待解决的问题，并指出了未来的各种方向。



【4】 Beyond Just Vision: A Review on Self-Supervised Representation Learning on Multimodal and Temporal Data<br>
**标题**：超越视觉：多通道和时态数据的自我监督表征学习研究综述<br>
**链接**：https://arxiv.org/abs/2206.02353<br>

**作者**：Shohreh Deldari,Hao Xue,Aaqib Saeed,Jiayuan He,Daniel V. Smith,Flora D. Salim<br>
**机构**： School of Computing and Technologies, RMIT University, School of Computer Science and Engineering, University of New South Wales<br>
**摘要**：近年来，自监督表征学习（SSRL）在计算机视觉、语音、自然语言处理（NLP）等领域引起了人们的广泛关注，近年来，它与其他类型的学习方式，包括传感器的时间序列，也引起了人们的广泛关注。自监督学习的流行是因为传统模型通常需要大量注释良好的数据进行训练。获取带注释的数据可能是一个困难且成本高昂的过程。引入了自监督方法，通过使用从原始数据中自由获得的监督信号对模型进行有区别的预训练，提高了训练数据的效率。与之前主要关注单一数据模式的CV或NLP领域的方法的SSRL现有综述不同，我们旨在首次全面综述时态数据的多模式自监督学习方法。为此，我们1）对现有SSRL方法进行全面分类，2）通过定义SSRL框架的关键组件引入通用管道，3）比较现有模型的目标功能、网络架构和潜在应用，以及4）审查每个类别和各种模式中的现有多模式技术。最后，我们提出了现有的弱点和未来的机会。我们相信，我们的工作为SSRL在使用多模态和/或时态数据的领域的需求提供了一个视角

<br><br>

【5】 Self-supervised Learning in Remote Sensing: A Review<br>
**标题**：遥感中的自我监督学习：综述<br>
**链接**：https://arxiv.org/abs/2206.13188<br>

**作者**：Yi Wang,Conrad M Albrecht,Nassim Ait Ali Braham,Lichao Mou,Xiao Xiang Zhu<br>
**摘要**：在深度学习研究中，自监督学习（SSL）受到了计算机视觉和遥感界的广泛关注。虽然计算机视觉已经取得了巨大的成功，但SSL在地球观测领域的大部分潜力仍然被锁定。在本文中，我们介绍并回顾了遥感背景下用于计算机视觉的SSL的概念和最新发展。此外，我们在流行的遥感数据集上提供了现代SSL算法的初步基准，验证了SSL在遥感中的潜力，并对数据扩充进行了扩展研究。最后，我们确定了SSL for earth observation（SSL4EO）未来研究的一系列有希望的方向，为这两个领域的有效互动铺平道路。

<br><br>

<a name="Zero/Few Shot迁移域适配自适应"/>

# **Zero/Few Shot|迁移|域适配|自适应**

【1】 A Brief Survey on Adaptive Video Streaming Quality Assessment<br>
**标题**：自适应视频流质量评估概述<br>
**链接**：https://arxiv.org/abs/2202.12987<br>

**作者**：Wei Zhou,Xiongkuo Min,Hong Li,Qiuping Jiang<br>
**摘要**：自适应视频流的体验质量（QoE）评估在先进的网络管理系统中发挥着重要作用。在HTTP上的动态自适应流媒体方案（DASH）中尤其具有挑战性，DASH具有越来越复杂的特性，包括额外的播放问题。在本文中，我们简要概述了自适应视频流质量评估。在回顾相关工作的基础上，我们分析和比较了在自适应视频流中使用或不使用机器学习技术的客观QoE评估模型的不同变化。通过性能分析，我们发现混合模型的性能优于服务质量（QoS）驱动的QoE方法和信号保真度测量。此外，基于机器学习的模型的性能略优于不使用机器学习的模型。此外，我们发现现有的视频流QoE评估模型性能有限，难以在实际通信系统中应用。因此，在传统视频质量预测的深度学习特征表示成功的基础上，我们还应用现成的深度卷积神经网络（DCNN）来评估流视频的感知质量，其中考虑了流视频的时空特性。实验证明了它的优越性，这为自适应视频流质量评估专门设计的深度学习框架的未来发展提供了启示。我们相信这项调查可以作为自适应视频流的QoE评估指南。

<br><br>

【2】 Learning from Few Examples: A Summary of Approaches to Few-Shot Learning<br>
**标题**：从几个例子中学习：几次学习方法综述<br>
**链接**：https://arxiv.org/abs/2203.04291<br>

**作者**：Archit Parnami,Minwoo Lee<br>
**机构**：Department of Computer Science, The University of North Carolina at Charlotte, Charlotte, NC, USA<br>
**摘要**：Few-Shot学习是指仅从几个训练样本中学习数据中的基本模式的问题。许多深度学习解决方案需要大量的数据样本，因此面临着数据饥饿和大量的计算时间和资源问题。此外，由于问题的性质或隐私问题，以及数据准备的成本，数据往往不可用。数据收集、预处理和标记是繁重的人工任务。因此，能够大幅缩短构建机器学习应用程序的周转时间的少量快照学习成为一种低成本的解决方案。这篇综述文章包含了最近提出的几种镜头学习算法的代表性列表。鉴于学习动态和特点，从元学习、迁移学习和混合方法（即少数镜头学习问题的不同变体）的角度讨论了解决少数镜头学习问题的方法。

<br><br>

<a name="点云SLAM雷达激光深度RGBD相关"/>

# **点云|SLAM|雷达|激光|深度RGBD相关**

【1】 Deep Depth Completion: A Survey<br>
**标题**：深井完井技术综述<br>
**链接**：https://arxiv.org/abs/2205.05335<br>

**作者**：Junjie Hu,Chenyu Bao,Mete Ozay,Chenyou Fan,Qing Gao,Honghai Liu,Tin Lun Lam<br>
**摘要**：深度完成的目的是从深度传感器获取的稀疏地图中预测密集的像素深度。它在自动驾驶、三维重建、增强现实和机器人导航等各种应用中发挥着重要作用。最近在这项任务上取得的成功已经被证明，并被基于深度学习的解决方案所主导。在本文中，我们首次提供了一个全面的文献综述，帮助读者更好地把握研究趋势，清楚地了解当前的进展。我们从网络架构、损失函数、基准数据集和学习策略的设计方面对相关研究进行了调查，并提出了一种新的分类法，对现有方法进行分类。此外，我们还对两个广泛使用的基准数据集（包括室内和室外数据集）的模型性能进行了定量比较。最后，我们讨论了之前工作的挑战，并为读者提供了一些关于未来研究方向的见解。

<br><br>

【2】 Surface Reconstruction from Point Clouds: A Survey and a Benchmark<br>
**标题**：基于点云数据的曲面重建：综述与基准<br>
**链接**：https://arxiv.org/abs/2205.02413<br>

**作者**：Zhangjin Huang,Yuxin Wen,Zihao Wang,Jinjuan Ren,Kui Jia<br>
**摘要**：从原始离散点云观测数据重建二维流形的连续曲面是一个长期存在的问题。该问题在技术上是不适定的，并且考虑到通过实际深度扫描获得的点云中会出现各种感测缺陷，该问题变得更加困难。在文献中，提出了一套丰富的方法，并对现有方法进行了综述。然而，现有的审查缺乏对共同基准的彻底调查。本论文旨在回顾和基准现有的方法在新时代的深度学习表面重建。为此，我们提供了一个由合成和真实扫描数据组成的大规模基准数据集；基准包括对象和场景级别的表面，并考虑了实际深度扫描中常见的各种感知缺陷。我们通过在构建的基准上比较现有方法进行了深入的实证研究，并特别关注现有方法对各种扫描缺陷的鲁棒性；我们还研究了不同方法在重建复杂曲面形状方面的通用性。我们的研究有助于确定不同方法工作的最佳条件，并提出一些实证结果。例如，虽然深度学习方法越来越流行，但我们的系统研究表明，令人惊讶的是，一些经典方法在稳健性和泛化方面表现得更好；我们的研究还表明，现有的所有曲面重建方法都没有解决多视点扫描中的点集错位、曲面点缺失和点异常等实际问题。我们希望基准和我们的研究对从业者和未来研究中的新创新具有价值。

<br><br>

【3】 Outdoor Monocular Depth Estimation: A Research Review<br>
**标题**：室外单目深度估计的研究进展<br>
**链接**：https://arxiv.org/abs/2205.01399<br>

**作者**：Pulkit Vyas,Chirag Saxena,Anwesh Badapanda,Anurag Goswami<br>
**机构**：School of Computer Science and Technology, Bennett University, Greater Noida, India<br>
**摘要**：深度估计是一项重要的任务，应用于计算机视觉的各种方法和应用中。虽然传统的深度估计方法基于深度提示，需要特定的设备，如立体摄像机，并根据所使用的方法配置输入，但目前的重点是单源或单目深度估计。卷积神经网络的最新发展，以及经典方法在这些深度学习方法中的集成，导致了深度估计问题的许多进展。户外深度估计问题，或野外深度估计问题，是一个研究非常少的领域。在本文中，我们概述了开放研究的可用数据集、深度估计方法、研究工作、趋势、挑战和机遇。据我们所知，没有公开可用的调查工作提供了户外深度估计技术和研究范围的全面集合，这使得我们的工作对希望进入这一研究领域的人做出了重要贡献。

<br><br>

【4】 Sequential Point Clouds: A Survey<br>
**标题**：连续点云：综述<br>
**链接**：https://arxiv.org/abs/2204.09337<br>

**作者**：Haiyan Wang,Yingli Tian<br>
**机构**： and the Department of Computer Science, the City University of New York<br>
**摘要**：点云已经引起了越来越多的研究和实际应用的关注。然而，其中许多应用（例如自动驾驶和机器人操作）实际上是基于连续点云（即四维），因为静态点云数据所能提供的信息仍然有限。最近，研究人员对连续点云进行了越来越多的研究。本文对基于深度学习的序列点云研究方法进行了广泛综述，包括动态流估计、目标检测与跟踪、点云分割和点云预测。本文进一步总结和比较了这些方法在公共基准数据集上的定量结果。最后，本文总结了当前序贯点云研究面临的挑战，并指出了未来的研究方向。

<br><br>

【5】 Comprehensive Review of Deep Learning-Based 3D Point Clouds Completion Processing and Analysis<br>
**标题**：基于深度学习的三维点云补全处理与分析综述<br>
**链接**：https://arxiv.org/abs/2203.03311<br>

**作者**：Ben Fei,Weidong Yang,Wenming Chen,Zhijun Li,Yikang Li,Tao Ma,Xing Hu,Lipeng Ma<br>
**机构**：Fudan University, Zhijun Li is with the Department of Automation, University of Science andTechnology of China<br>
**摘要**：点云完备性是由局部点云衍生而来的一个生成和估计问题，在三维计算机视觉应用中起着至关重要的作用。深度学习（DL）的进展显著提高了点云完成的能力和鲁棒性。然而，完成的点云的质量仍需进一步提高，以满足实际应用。因此，这项工作旨在对各种方法进行全面调查，包括基于点、基于卷积、基于图形和基于生成模型的方法等。本调查总结了这些方法之间的比较，以激发进一步的研究见解。此外，本文还总结了常用的数据集，并举例说明了点云技术的应用。最后，我们还讨论了这个迅速扩展的领域可能的研究趋势。


<br><br>

<a name="3D3D重建"/>

# **3D|3D重建**

【1】 A Survey of Non-Rigid 3D Registration<br>
**标题**：非刚体三维配准技术综述<br>
**链接**：https://arxiv.org/abs/2203.07858<br>

**作者**：Bailin Deng,Yuxin Yao,Roberto M. Dyke,Juyong Zhang<br>
**机构**：†, Cardiff University, University of Science and Technology of China, Università della Svizzera italiana<br>
**摘要**：非刚性配准以非刚性方式计算源曲面与目标曲面之间的对齐。在过去的十年中，随着能够测量时变表面的三维传感技术的进步，非刚性配准已被应用于可变形形状的获取，并有着广泛的应用。本文综述了三维形状的非刚性配准方法，重点介绍了与动态形状获取和重建相关的技术。特别是，我们回顾了表示变形场的不同方法，以及计算所需变形的方法。包括基于优化和基于学习的方法。我们还回顾了评估非刚性配准方法的基准和数据集，并讨论了未来可能的研究方向。

<br><br>

<a name="Attention注意力"/>

# Attention注意力

【1】 Visual Attention Methods in Deep Learning: An In-Depth Survey<br>
**标题**：深度学习中的视觉注意方法：一项深入调查<br>
**链接**：https://arxiv.org/abs/2204.07756<br>

**作者**：Mohammed Hassanin,Saeed Anwar,Ibrahim Radwan,Fahad S Khan,Ajmal Mian<br>
**机构**： and The University of Technology Sydney, •Ibrahim Radwan is with University of Canberra, Khan is an Associate Professor with Mohammad Bin ZayedUniversity of Artificial Intelligence<br>
**摘要**：受人类认知系统的启发，注意力是一种机制，它模仿人类对特定信息的认知意识，放大关键细节，从而更加关注数据的基本方面。深度学习利用注意力提高了许多应用程序的性能。有趣的是，同样的注意力设计可以适合处理不同的数据模式，并且可以很容易地整合到大型网络中。此外，多个互补注意机制可以整合到一个网络中。因此，注意力技巧变得非常有吸引力。然而，文献中缺乏针对注意技巧的全面调查，以指导研究人员在深层模型中使用注意。请注意，除了在训练数据和计算资源方面要求很高之外，《Transformer》在众多类别中只涵盖了一个类别的自我关注。我们填补了这一空白，并对50种注意力技巧进行了深入调查，根据它们最显著的特征对它们进行了分类。我们首先介绍了注意力机制成功背后的基本概念。接下来，我们将提供一些要点，例如每个注意力类别的优势和局限性，描述它们的基本构造块、主要用途的基本公式，以及专门用于计算机视觉的应用。我们还讨论了与一般注意机制相关的挑战和开放性问题。最后，我们建议未来可能的研究方向，以引起高度关注。

<br><br>

<a name="裁剪量化加速压缩相关"/>

# **裁剪|量化|加速|压缩相关**

【1】 Dimensionality Reduced Training by Pruning and Freezing Parts of a Deep Neural Network, a Survey<br>
**标题**：深度神经网络部分剪枝和冻结降维训练研究综述<br>
**链接**：https://arxiv.org/abs/2205.08099<br>

**作者**：Paul Wimmer,Jens Mehnert,Alexandru Paul Condurache<br>
**机构**：a,Robert Bosch GmbH, Automated Driving Research, Burgenlandstrasse , Stuttgart, Germany, University of L¨ubeck, Institute for Signal Processing, Ratzeburger Allee , L¨ubeck, Germany<br>
**摘要**：最先进的深度学习模型的参数计数高达数十亿。训练、存储和传输此类模型需要耗费大量精力和时间，因此成本高昂。这些成本的很大一部分是由网络训练引起的。模型压缩降低了存储和传输成本，并可以通过减少向前和/或向后传递中的计算数量进一步提高训练效率。因此，在保持高性能的同时也在训练时压缩网络是一个重要的研究课题。这项工作是对在整个训练过程中减少深度学习模型中训练权重的方法的调查。大多数引入的方法将网络参数设置为零，这称为剪枝。提出的剪枝方法分为初始化剪枝、彩票剪枝和动态稀疏训练剪枝。此外，我们还讨论了在随机初始化时冻结部分网络的方法。通过冻结权重，减少了可训练参数的数量，从而减少了梯度计算和模型优化空间的维数。在这项调查中，我们首先提出降维训练作为一个基本的数学模型，涵盖了训练期间的修剪和冻结。然后，我们提出并讨论了不同的降维训练方法。

<br><br>

<a name="语义分割"/>

# 语义分割

【1】 Semantic Segmentation for Thermal Images: A Comparative Survey<br>
**标题**：热图像语义分割的比较研究<br>
**链接**：https://arxiv.org/abs/2205.13278<br>

**作者**：Zülfiye Kütük,Görkem Algan<br>
**摘要**：语义分割是一项具有挑战性的任务，因为与其他计算机视觉问题相比，它需要更多的图像低层次空间信息。像素级分类的准确性会受到许多因素的影响，例如图像限制和图像中对象边界的模糊性。传统的方法利用深度神经网络（DNN）在可见光谱中捕获的三通道RGB图像。热图像可以在分割过程中发挥重要作用，因为热成像相机能够捕捉细节，无论天气和照明条件如何。在语义分割中使用红外光谱有许多实际用例，如自动驾驶、医疗成像、农业、国防工业等。由于这些用例的广泛性，利用红外光谱设计准确的语义分割算法是一个重要的挑战。一种方法是使用可见光和红外光谱图像作为输入。由于丰富的输入信息，这些方法可以实现更高的精度，同时需要花费额外的精力来对齐和处理多个输入。另一种方法是只使用热图像，以便在较小的用例中实现较低的硬件成本。尽管有很多关于语义分割方法的调查，但文献中缺乏一个明确围绕使用红外光谱进行语义分割的全面调查。这项工作旨在通过在文献中介绍算法并根据输入图像对其进行分类来填补这一空白。

<br><br>

【2】 On Efficient Real-Time Semantic Segmentation: A Survey<br>
**标题**：高效实时语义切分研究综述<br>
**链接**：https://arxiv.org/abs/2206.08605<br>

**作者**：Christopher J. Holder,Muhammad Shafique<br>
**备注**：18 pages, 13 figures, 4 tables This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible<br>
**摘要**：语义分割是为图像中的每个像素指定一个类别标签的问题，是自动车辆视觉堆栈的重要组成部分，有助于场景理解和目标检测。然而，许多性能最好的语义分割模型极其复杂和繁琐，因此不适合部署在计算资源有限且对低延迟操作至关重要的车载自主车辆平台上。在本次调查中，我们深入研究了旨在通过更紧凑和高效的模型来解决这种错位的工作，这些模型能够部署在低内存嵌入式系统上，同时满足实时推理的约束。我们讨论了该领域中几个最突出的工作，根据它们的主要贡献将它们放在一个分类中，最后我们评估了在一致的硬件和软件设置下所讨论模型的推理速度，这些硬件和软件设置代表了具有高端GPU的典型研究环境和使用低内存嵌入式GPU硬件的实际部署场景。我们的实验结果表明，许多工作能够在资源受限的硬件上实现实时性能，同时说明了延迟和准确性之间的一致性权衡。

<br><br>

【3】 A Survey on Deep Learning for Skin Lesion Segmentation<br>
**标题**：深度学习在皮肤病变分割中的研究进展<br>
**链接**：https://arxiv.org/abs/2206.00356<br>

**作者**：Zahra Mirikharaji,Catarina Barata,Kumar Abhishek,Alceu Bissoto,Sandra Avila,Eduardo Valle,M. Emre Celebi,Ghassan Hamarneh<br>
**机构**：Medical Image Analysis Lab, School of Computing Science, Simon Fraser University, Burnaby V,A ,S, Canada, Institute for Systems and Robotics, Instituto Superior T´ecnico, Avenida Rovisco Pais, Lisbon ,-, Portugal<br>
**摘要**：皮肤癌是一个主要的公共卫生问题，可以通过计算机辅助诊断来减轻这种常见疾病的负担。从图像中分割皮肤损伤是实现这一目标的重要步骤。然而，自然和人工伪影（如头发和气泡）、内在因素（如病变形状和对比度）以及图像采集条件的变化使皮肤病变分割成为一项具有挑战性的任务。最近，许多研究人员探索了深度学习模型在皮肤损伤分割中的适用性。在这项调查中，我们交叉研究了134篇涉及基于深度学习的皮肤病变分割的研究论文。我们从多个维度分析这些工作，包括输入数据（数据集、预处理和合成数据生成）、模型设计（架构、模块和损失）和评估方面（数据注释要求和分段性能）。我们从精选的开创性作品和系统的角度讨论这些维度，考察这些选择如何影响当前的趋势，以及应如何解决其局限性。我们在一个综合表格中总结了所有已检查的工作，以便于比较。

<br><br>

<a name="其他神经网络深度学习模型建模"/>

# **其他神经网络|深度学习|模型|建模**

【1】 Machine Learning-based Biological Ageing Estimation Technologies: A Survey<br>
**标题**：基于机器学习的生物老化评估技术综述<br>
**链接**：https://arxiv.org/abs/2206.12650<br>

**作者**：Zhaonian Zhang,Richard Jiang,Danny Crookes,Paul Chazot<br>
**摘要**：近年来，人们发展了各种各样的估计生物年龄的方法。特别是随着机器学习（ML）的发展，BA预测的类型越来越多，准确率也有了很大的提高。BA估计模型在监测健康老龄化方面发挥着重要作用，可以为检测一般人群的健康状况和向亚健康人群发出警告提供新的工具。我们将主要回顾三种基于血液生物标记物、面部图像和结构神经成像特征的年龄预测方法。目前，使用血液生物标记物的模型是最简单、最直接和最准确的方法。人脸图像方法受种族、环境等多方面的影响，预测精度不是很高，不能对医学领域做出很大贡献。总之，我们在这里为我们和其他潜在的普通人群跟踪大数据时代的前进方向，并展示如何利用当今的大量可用数据。

<br><br>

【2】 Review Neural Networks about Image Transformation Based on IGC Learning Framework with Annotated Information<br>
**标题**：基于带注释信息的IGC学习框架的图像变换神经网络综述<br>
**链接**：https://arxiv.org/abs/2206.10155<br>

**作者**：Yuanjie Yan,Suorong Yang,Yan Wang,Jian Zhao,Furao Shen<br>
**摘要**：图像变换是一类以学习输入图像和输出图像之间的映射为目标的视觉和图形问题，在深度神经网络的背景下发展迅速。在计算机视觉中，许多问题都可以看作是图像转换的任务，例如语义分割和风格转换。这些作品有着不同的主题和动机，使得图像转换任务蓬勃发展。一些调查只回顾了风格转换或意象翻译的研究，这些都只是意象转换的一个分支。然而，据我们所知，没有一项调查在一个统一的框架内总结这些工作。本文提出了一种新的学习框架，包括自主学习、引导学习和合作学习，称为IGC学习框架。我们讨论的图像转换主要涉及深度神经网络的一般图像到图像的转换和风格转换。从这个框架的角度来看，我们审查了这些子任务，并对各种场景给出了统一的解释。我们根据相似的发展趋势对图像变换的相关子任务进行分类。此外，还通过实验验证了IGC学习的有效性。最后，讨论了新的研究方向和有待进一步研究的问题。

<br><br>

【3】 A Survey of Automated Data Augmentation Algorithms for Deep Learning-based Image Classication Tasks<br>
**标题**：基于深度学习的图像分类任务自动数据增强算法综述<br>
**链接**：https://arxiv.org/abs/2206.06544<br>

**作者**：Zihan Yang,Richard O. Sinnott,James Bailey,Qiuhong Ke<br>
**机构**：The, University of Melbourne, Swanston Street, Melbourne, Victoria, Australia., Corresponding author(s). <br>

**摘要**：近年来，计算机视觉领域最流行的技术之一就是深度学习技术。作为一种数据驱动技术，深度模型需要大量精确标记的训练数据，而在许多实际应用中，这些数据往往无法访问。数据空间解决方案是数据增强（DA），它可以从原始样本中人工生成新图像。图像增强策略可能因数据集而异，因为不同的数据类型可能需要不同的增强来促进模型训练。然而，DA策略的设计在很大程度上是由具有领域知识的人类专家决定的，这被认为是非常主观和容易出错的。为了缓解这种问题，一个新的方向是使用自动数据增强（AutoDA）技术从给定的数据集中自动学习图像增强策略。AutoDA模型的目标是找到能够最大化模型性能增益的最佳DA策略。本文从图像分类的角度探讨了AutoDA技术出现的根本原因。我们确定了标准AutoDA模型的三个关键组件：搜索空间、搜索算法和评估函数。基于它们的体系结构，我们提供了现有图像AutoDA方法的系统分类。本文介绍了AutoDA领域的主要工作，讨论了它们的优缺点，并提出了未来改进的几个潜在方向。

<br><br>

【4】 Applications of Deep Learning in Fish Habitat Monitoring: A Tutorial and Survey<br>
**标题**：深度学习在鱼类生境监测中的应用：教程与综述<br>
**链接**：https://arxiv.org/abs/2206.05394<br>

**作者**：Alzayat Saleh,Marcus Sheaves,Dean Jerry,Mostafa Rahimi Azghadi<br>
**摘要**：海洋生态系统及其鱼类栖息地在提供宝贵的食物来源和保护成果方面发挥着不可或缺的作用，因此变得越来越重要。由于海洋环境和鱼类栖息地偏远且难以接近，因此通常使用水下摄像头对其进行监测。这些相机产生大量的数字数据，而当前的手动处理方法无法有效地分析这些数据，因为这些方法需要一个人来观察。DL是一种前沿人工智能技术，在分析视觉数据方面表现出前所未有的性能。尽管其应用于无数领域，但其在水下鱼类栖息地监测中的应用仍有待探索。在本文中，我们提供了一个教程，介绍DL的关键概念，帮助读者从较高的层次理解DL的工作原理。本教程还解释了一步一步的过程，说明了如何为具有挑战性的应用程序（如水下鱼类监测）开发DL算法。此外，我们还全面调查了鱼类栖息地监测的关键深度学习技术，包括分类、计数、定位和分割。此外，我们调查了公开的水下鱼类数据集，并比较了水下鱼类监测领域中的各种DL技术。我们还讨论了鱼类栖息地处理深度学习新兴领域的一些挑战和机遇。本文旨在为海洋科学家提供一个教程，他们希望掌握DL的高级理解，通过遵循我们的分步教程为其应用开发DL，并了解其如何发展以促进其研究工作。同时，它适用于希望调查基于DL的最新鱼类栖息地监测方法的计算机科学家。

<br><br>

【5】 Learning with Capsules: A Survey<br>
**标题**：胶囊学习：综述<br>
**链接**：https://arxiv.org/abs/2206.02664<br>

**摘要**：胶囊网络被提出作为卷积神经网络（CNN）的一种替代方法，用于学习以对象为中心的表示，可用于改进泛化和样本复杂性。与CNN不同，胶囊网络被设计为通过使用神经元组对视觉实体进行编码，并学习这些实体之间的关系，来显式地建模部分-整体层次关系。capsule networks早期取得的良好成果促使深度学习社区继续努力提高其在多个应用领域的性能和可扩展性。然而，胶囊网络研究的一个主要障碍是缺乏一个可靠的参考点来理解他们的基本想法和动机。本次调查的目的是全面概述capsule network研究领域，这将成为社区未来的宝贵资源。为此，我们首先介绍胶囊网络背后的基本概念和动机，例如计算机视觉中的等变推理。然后，我们介绍了胶囊路由机制的技术进展以及胶囊网络的各种公式，例如生成和几何。此外，我们还详细解释了胶囊网络与Transformer中流行的注意机制之间的关系，并强调了在表征学习的背景下它们之间的非平凡的概念相似性。之后，我们探讨了胶囊网络在计算机视觉、视频和运动、图形表示学习、自然语言处理、医学成像等领域的广泛应用。最后，我们对胶囊网络研究中的主要障碍进行了深入讨论，并强调了未来工作的研究方向。

<br><br>

【6】 A Review of Published Machine Learning Natural Language Processing Applications for Protocolling Radiology Imaging<br>
**标题**：已发表的机器学习自然语言处理在放射成像中的应用综述<br>
**链接**：https://arxiv.org/abs/2206.11502<br>

**作者**：Nihal Raju,Michael Woodburn,Stefan Kachel,Jack O'Shaughnessy,Laurence Sorace,Natalie Yang,Ruth P Lim<br>

**摘要**：机器学习（ML）是人工智能（AI）的一个分支领域，其在放射学中的应用正以前所未有的速度增长。研究最多的ML应用是图像的自动解释。然而，自然语言处理（NLP）可以与ML结合用于文本解释任务，在放射学中也有许多潜在的应用。其中一个应用是放射科协议的自动化，它涉及到解释临床放射科转诊和选择适当的成像技术。这是确保正确成像的一项基本任务。然而，放射科医生必须致力于protocolling的时间可能会花在报告、与推荐人沟通或教学上。迄今为止，很少有出版物开发了使用临床文本自动选择方案的ML模型。本文回顾了该领域的现有文献。参考机器学习公约建议的最佳实践，对已发布的模型进行系统评估。讨论了在临床环境中实现自动预冷的进展。

<br><br>

<a name="蒸馏知识提取"/>

# **蒸馏|知识提取**

【1】 Backbones-Review: Feature Extraction Networks for Deep Learning and Deep Reinforcement Learning Approaches<br>
**标题**：支持深度学习的特征提取网络和深度强化学习方法综述<br>
**链接**：https://arxiv.org/abs/2206.08016<br>

**作者**：Omar Elharroussad,Younes Akbari,Noor Almaadeed,Somaya Al-Maadeed<br>
**机构**：Department of Computer Science and Engineering, Qatar University,Doha, Qatar, A R T I C L E I N F O<br>
**摘要**：为了使用各种类型的数据来理解现实世界，人工智能（AI）是当今最常用的技术。而在分析的数据中找到模式则是主要任务。这是通过提取代表性特征步骤来执行的，该步骤使用统计算法或一些特定的过滤器进行。然而，从大规模数据中选择有用的特征是一个至关重要的挑战。现在，随着卷积神经网络（CNN）的发展，特征提取操作变得更加自动化和简单。CNN允许处理大规模数据，并覆盖特定任务的不同场景。对于计算机视觉任务，卷积网络也用于提取深度学习模型其他部分的特征。为特征提取或DL模型的其他部分选择合适的网络不是随机工作。因此，这种模型的实现可能与目标任务及其计算复杂性有关。许多网络已经被提出，并成为任何人工智能任务中用于任何DL模型的著名网络。这些网络可用于特征提取或在任何DL模型（称为主干）的开始处使用。主干网是以前在许多其他任务中受过训练的已知网络，并证明其有效性。本文概述了现有主干网，如VGG、RESNET、DenseNet等，并对其进行了详细描述。此外，还讨论了几个计算机视觉任务，回顾了每个任务所使用的主干。此外，还根据每个任务使用的主干网，对性能进行了比较。

<br><br>

<a name="其他"/>

# **其他**

【1】 Graph Neural Networks: a bibliometrics overview<br>
**标题**：图神经网络：文献计量学综述<br>
**链接**：https://arxiv.org/abs/2201.01188<br>

**作者**：Abdalsamad Keramatfar,Mohadeseh Rafiee,Hossein Amirkhani<br>
**机构**： SID, Academic Center for Education, Culture and Research (ACECR( , Tehran, Iran, Department of Computer Engineering and IT, University of Qom, Qom, Iran<br>
**摘要**：近年来，图神经网络已成为机器学习领域的一个研究热点。本文介绍了自2004年GNN论文首次发表以来，基于Scopus的GNN研究文献计量学概述。本研究旨在定量和定性地评估GNN的研究趋势。我们提供研究趋势、主题分布、活跃和有影响力的作者和机构、出版物来源、被引用最多的文献和热门话题。我们的调查显示，该领域最常见的学科类别是计算机科学、工程、电信、语言学、运筹学和管理科学、信息科学和图书馆学、商业和经济学、自动化和控制系统、机器人学和社会科学。此外，GNN出版物最活跃的来源是计算机科学的课堂讲稿。美国、中国和加拿大的机构数量最多，影响最大。我们还提供必读论文和未来方向。最后，图卷积网络的应用和注意机制是GNN研究的热点。

<br><br>

【2】 Scene Graph Generation: A Comprehensive Survey<br>
**标题**：场景图生成：综述<br>
**链接**：https://arxiv.org/abs/2201.00443<br>

**作者**：Guangming Zhu,Liang Zhang,Youliang Jiang,Yixuan Dang,Haoran Hou,Peiyi Shen,Mingtao Feng,Xia Zhao,Qiguang Miao,Syed Afaq Ali Shah,Mohammed Bennamoun<br>
**机构**： Zhao is with the School of Arts and Sciences, National University ofDefense Technology<br>
**摘要**：近年来，深度学习技术在通用目标检测领域取得了显著突破，并催生了大量场景理解任务。场景图以其强大的语义表示和在场景理解中的应用而成为研究的热点。场景图生成（scenegraphgeneration，SGG）是指将图像自动映射为语义结构场景图的任务，这需要对检测到的对象及其关系进行正确的标记。尽管这是一项具有挑战性的任务，但社区已经提出了许多SGG方法，并取得了良好的效果。在这篇文章中，我们提供了一个由深度学习技术在这一领域的最新成就的综合调查。我们回顾了138篇涉及不同输入模式的代表性著作，并从特征提取和融合的角度系统总结了现有的基于图像的SGG方法。我们试图将现有的视觉关系检测方法联系起来并加以系统化，以全面的方式总结和解释SGG的机制和策略。最后，我们对当前存在的问题和未来的研究方向进行了深入的讨论。这项调查将有助于读者更好地了解当前的研究现状和想法。

<br><br>

【3】 Visual and Object Geo-localization: A Comprehensive Survey<br>
**标题**：视觉与物体地理定位：综述<br>
**链接**：https://arxiv.org/abs/2112.15202<br>

**作者**：Daniel Wilson,Xiaohan Zhang,Waqas Sultani,Safwan Wshah<br>
**摘要**：地理定位的概念是指确定某些“实体”在地球上的位置的过程，通常使用全球定位系统（GPS）坐标。感兴趣的实体可以是图像、图像序列、视频、卫星图像，甚至是图像内可见的对象。由于智能手机和互联网的出现，GPS标记媒体的海量数据集迅速可用，而深度学习已经兴起，以增强机器学习模型的性能，视觉和物体地理定位领域的出现是因为它对增强现实、机器人技术、自动驾驶车辆、道路维护和三维重建等广泛应用产生了重大影响。在涉及地理定位的图纸中，确定地理对象的综合定位（包括地理定位图像或地理定位图像）。我们将提供深入的研究，包括对流行算法的总结、对提议的数据集的描述以及对性能结果的分析，以说明每个领域的当前状态。

<br><br>

【4】 Data-Free Knowledge Transfer: A Survey<br>
**标题**：无数据知识转移：综述<br>
**链接**：https://arxiv.org/abs/2112.15278<br>

**作者**：Yuang Liu,Wei Zhang,Jun Wang,Jianyong Wang<br>
**摘要**：在过去的十年中，许多深度学习模型得到了很好的训练，并在机器智能的各个领域取得了巨大的成功，特别是在计算机视觉和自然语言处理领域。为了更好地利用这些训练有素的模型在领域内或跨领域迁移学习中的潜力，知识提取（KD）和领域适应（DA）被提出并成为研究热点。它们的目的都是从经过良好训练的模型和原始训练数据中传递有用的信息。然而，由于隐私、版权或保密性的原因，原始数据在许多情况下并不总是可用的。最近，无数据知识转移范式引起了广泛关注，因为它涉及从训练有素的模型中提取有价值的知识，而无需访问训练数据。特别是，它主要包括数据自由知识提取（DFKD）和源数据自由域自适应（SFDA）。一方面，DFKD的目标是将原始数据的域内知识从繁琐的教师网络转移到紧凑的学生网络，以实现模型压缩和高效推理。另一方面，SFDA的目标是重用存储在经过良好训练的源模型中的跨领域知识，并使其适应目标领域。本文从知识提炼和无监督领域适应两个角度对数据自由知识转移进行了全面的综述，以帮助读者更好地了解当前的研究现状和思路。简要回顾了这两个领域的应用和面临的挑战。此外，我们还为未来的研究提供了一些见解。


<br><br>

【5】 Deep Learning for Omnidirectional Vision: A Survey and New Perspectives<br>
**标题**：全方位视觉的深度学习：综述与新视角<br>
**链接**：https://arxiv.org/abs/2205.10468<br>

**作者**：Hao Ai,Zidong Cao,Jinjing Zhu,Haotian Bai,Yucheng Chen,Ling Wang<br>
**机构**： The Hong Kong University of Science and Technology (HKUST)<br>
**摘要**：全向图像（ODI）数据是用360x180的视场拍摄的，该视场比针孔相机宽得多，并且包含比传统平面图像更丰富的空间信息。因此，全方位视觉因其在自主驾驶和虚拟现实等众多应用中具有更优越的性能而备受关注。近年来，客户级360摄像头的可用性使得全方位视觉更加流行，而深度学习（DL）的发展极大地激发了其研究和应用。本文对全方位视觉DL方法的最新进展进行了系统、全面的回顾和分析。我们的工作包括四个主要内容：（i）介绍了全方位成像的原理、ODI上的卷积方法以及与二维平面图像数据相比的差异和困难的数据集；（ii）全方位视觉DL方法的结构和层次分类；（iii）总结最新的学习策略和应用；（iv）通过突出潜在的研究方向，对挑战和开放问题进行深入讨论，以触发社区中的更多研究。

<br><br>

【6】 Deep Learning for Visual Speech Analysis: A Survey<br>
**标题**：深度学习在视觉语音分析中的研究进展<br>
**链接**：https://arxiv.org/abs/2205.10839<br>

**作者**：Changchong Sheng,Gangyao Kuang,Liang Bai,Chenping Hou,Yulan Guo,Xin Xu,Matti Pietikäinen,Li Liu<br>
**摘要**：视觉言语是指言语的视觉领域，由于其在公安、医疗、军事防御、电影娱乐等领域的广泛应用，越来越受到人们的关注。深度学习技术作为一种强大的人工智能策略，广泛地促进了视觉语音学习的发展。在过去的五年中，人们提出了许多基于深度学习的方法来解决这一领域的各种问题，尤其是自动视觉语音识别和生成。为了推进视觉语音的未来研究，本文旨在全面回顾视觉语音分析深度学习方法的最新进展。我们涵盖了可视语音的不同方面，包括基本问题、挑战、基准数据集、现有方法的分类以及最先进的性能。此外，我们还发现了当前研究中的差距，并讨论了启发未来研究的方向。

<br><br>

【7】 Hyperspectral Unmixing Based on Nonnegative Matrix Factorization: A Comprehensive Review<br>
**标题**：基于非负矩阵分解的高光谱分解：综述<br>
**链接**：https://arxiv.org/abs/2205.09933<br>

**作者**：Xin-Ru Feng,Heng-Chao Li,Rui Wang,Qian Du,Xiuping Jia,Antonio Plaza<br>
**机构**： Southwest Jiaotong University<br>
**摘要**：高光谱分解是一种重要的技术，它可以从高光谱图像（HSI）中估计一组端元及其相应的丰度。非负矩阵分解（NMF）在解决这一问题中发挥着越来越重要的作用。在这篇文章中，我们对基于NMF的高光谱分解方法进行了全面的综述。以NMF模型为基准，我们展示了如何利用HSI的主要特性（例如，光谱、空间和结构信息）改进NMF。我们将约束NMF、结构化NMF和广义NMF分为三个重要的发展方向。此外，还进行了一些实验来说明相关算法的有效性。最后，我们对文章进行了总结，提出了未来可能的发展方向，旨在为促进高光谱分解的发展提供指导和启示。

<br><br>

【8】 Tensor Decompositions for Hyperspectral Data Processing in Remote Sensing: A Comprehensive Review<br>
**标题**：张量分解在高光谱遥感数据处理中的应用综述<br>
**链接**：https://arxiv.org/abs/2205.06407<br>

**作者**：Minghua Wang,Danfeng Hong,Zhu Han,Jiaxin Li,Jing Yao,Lianru Gao,Bing Zhang,Jocelyn Chanussot<br>
**机构**： Han is with the Key Laboratory of Digital Earth Science, Aerospace In-formation Research Institute<br>
**摘要**：由于传感器技术的快速发展，高光谱遥感（HS）成像为在飞机、航天器和卫星等数据采集设备的距离上观测和分析地球表面提供了大量的空间和光谱信息。HS-RS技术的最新进展甚至革命为充分发挥各种应用的潜力提供了机会，同时也为高效处理和分析巨大的HS采集数据带来了新的挑战。在过去的几十年里，张量分解由于保持了三维HS的固有结构，在HS数据处理任务中引起了广泛的关注和研究。在这篇文章中，我们旨在全面概述张量分解，特别是结合HS数据处理中的五大主题，即HS恢复、压缩感知、异常检测、超分辨率和光谱分解。对于每一个主题，我们详细介绍了HS-RS张量分解模型的显著成就，对现有方法进行了关键性描述，并对实验结果进行了代表性展示。因此，本文从实际的HS-RS实践和张量分解与高级先验甚至深度神经网络相结合的角度概述和讨论了后续研究方向的剩余挑战。本文总结了不同的基于张量分解的HS数据处理方法，并将其分为不同的类别，从简单的采用到算法初学者与其他先验知识的复杂组合。我们也期望这项调查能够为在一定程度上理解张量分解和HS-RS的有经验的研究人员提供新的研究和发展趋势。

<br><br>

【9】 Deep Learning and Computer Vision Techniques for Microcirculation Analysis: A Review<br>
**标题**：深度学习和计算机视觉微循环分析技术综述<br>
**链接**：https://arxiv.org/abs/2205.05493<br>

**作者**：Maged Abdalla Helmy Mohamed Abdou,Trung Tuyen Truong,Eric Jul,Paulo Ferreira<br>
**摘要**：微循环图像分析有可能揭示败血症等危及生命的疾病的早期迹象。微循环图像中毛细血管密度和分布的量化可以作为一种生物标志物来帮助危重病人。这些生物标记物的量化是劳动密集型、耗时的，并且受制于观察者之间的差异。根据上述挑战，几种性能各异的计算机视觉技术可用于自动分析这些微循环图像。在本文中，我们对50多篇研究论文进行了综述，并介绍了最相关和最有前途的计算机视觉算法，以自动分析微循环图像。此外，我们还介绍了其他研究人员目前用来自动分析微循环图像的方法。这项调查具有很高的临床相关性，因为它为其他研究人员开发微循环分析系统和算法提供了技术指南。

<br><br>

【10】 A Comprehensive Survey of Image Augmentation Techniques for Deep Learning<br>
**标题**：面向深度学习的图像增强技术综述<br>
**链接**：https://arxiv.org/abs/2205.01491<br>

**作者**：Mingle Xu,Sook Yoon,Alvaro Fuentes,Dong Sun Park<br>
**机构**：Department of Electronics Engineering, Jeonbuk National University, Jeonbuk , South, Department of Computer Engineering, Mokpo National University, Jeonnam , South, Core Research Institute of Intelligent Robots, Jeonbuk National University, Jeonbuk<br>
**摘要**：深度学习在需要大量图像的计算机视觉中取得了良好的性能，然而，在许多情况下，收集图像既昂贵又困难。为了缓解这个问题，许多图像增强算法被认为是有效的策略。了解当前的算法对于为给定任务找到合适的方法或开发新技术至关重要。在本文中，我们使用一种新的信息分类法对用于深度学习的图像增强进行了全面的调查。为了了解为什么我们需要图像增强，我们介绍了计算机视觉任务和邻近分布中的挑战。然后，将算法分为三类；无模型、基于模型和基于策略的优化。无模型类别采用图像处理方法，而基于模型的方法利用可训练的图像生成模型。相比之下，基于优化策略的方法旨在找到最优操作或它们的组合。此外，我们还讨论了两个更为活跃的主题，即利用不同的方法来理解图像增强，如群理论和核理论，以及将图像增强用于无监督学习的当前趋势。基于这些分析，我们相信我们的调查提供了一个更好的理解，有助于选择合适的方法或为实际应用设计新的算法。

<br><br>

【11】 Vision-and-Language Pretrained Models: A Survey<br>
**标题**：视觉和语言预训练模型：综述<br>
**链接**：https://arxiv.org/abs/2204.07356<br>

**作者**：Siqu Long,Feiqi Cao,Soyeon Caren Han,Haiqing Yang<br>
**机构**：School of Computer Science, The University of Sydney, Australia, CTO Lab, International Digital Economy Academy, China<br>
**摘要**：预训练模型在计算机视觉（CV）和自然语言处理（NLP）方面都取得了巨大的成功。这一进展通过将视觉和语言内容输入到一个多层转换器，即视觉语言预训练模型（VLPM），从而学习视觉和语言预训练的联合表征。在本文中，我们概述了VLPM在视觉和语言联合表达方面取得的主要进展。作为开场白，我们简要介绍了VLPM的一般任务定义和遗传结构。我们首先讨论了语言和视觉数据的编码方法，然后介绍了主流的VLPM结构作为核心内容。我们进一步总结了几个基本的预训练和微调策略。最后，我们强调了CV和NLP研究人员的三个未来方向，以提供有见地的指导。

<br><br>

【12】 Interpretability of Machine Learning Methods Applied to Neuroimaging<br>
**标题**：应用于神经成像的机器学习方法的可解释性<br>
**链接**：https://arxiv.org/abs/2204.07005<br>

**作者**：Elina Thibeau-Sutre,Sasha Collin,Ninon Burgos,Olivier Colliot<br>
**摘要**：深度学习方法在自然图像处理中非常流行，并成功地应用于神经成像领域。由于这些方法是不透明的，因此需要使用可解释性方法来验证它们并确保其可靠性。事实上，研究表明，通过利用训练集中的偏差，即使使用不相关的特征，深度学习模型也可以获得较高的性能。通过使用可解释性方法，可以潜在地检测到这种不良情况。最近，人们提出了许多方法来解释神经网络。然而，这个领域还不成熟。机器学习用户在试图解释他们的模型时面临两个主要问题：选择哪种方法，以及如何评估其可靠性？在这里，我们的目标是通过介绍最常见的可解释性方法和指标来提供这些问题的答案，这些方法和指标用于评估它们的可靠性，以及它们在神经成像环境中的应用和基准。请注意，这并不是一项详尽的调查：我们的目标是专注于我们发现最具代表性和相关性的研究。

<br><br>

【13】 DL4SciVis: A State-of-the-Art Survey on Deep Learning for Scientific Visualization<br>
**标题**：DL4SciVis：科学可视化深度学习的最新研究进展<br>
**链接**：https://arxiv.org/abs/2204.06504<br>

**作者**：Chaoli Wang,Jun Han<br>
**机构**： Han are with the Department of Computer Science andEngineering, University of Notre Dame<br>
**摘要**：自2016年以来，我们见证了人工智能+可视化（AI+VIS）研究的巨大发展。然而，关于AI+VIS的现有调查论文侧重于视觉分析和信息可视化，而不是科学可视化（SciVis）。在本文中，我们调查了SciVis中的相关深度学习（DL）工作，特别是在DL4SciVis的方向：为解决SciVis问题设计DL解决方案。为了保持专注，我们主要考虑处理标量和向量场数据但不包括网格数据的工作。我们从六个维度对这些工作进行分类和讨论：领域设置、研究任务、学习类型、网络结构、损失函数和评估指标。本文最后讨论了需要填补的剩余差距，以及我们作为一个社区需要解决的重大挑战。这项最先进的调查指导SciVis研究人员对这一新兴主题进行概述，并指出未来发展这项研究的方向。

<br><br>

【14】 Biclustering Algorithms Based on Metaheuristics: A Review<br>
**标题**：基于元启发式的双聚类算法综述<br>
**链接**：https://arxiv.org/abs/2203.16241<br>

**作者**：Adan Jose-Garcia,Julie Jacques,Vincent Sobanski,Clarisse Dhaenens<br>
**摘要**：双聚类是一种无监督的机器学习技术，可以同时对数据矩阵中的行和列进行聚类。双聚类已成为一种重要的方法，并在生物信息学、文本挖掘和模式识别等各种应用中发挥着重要作用。然而，寻找重要的双聚类是一个NP难问题，可以表述为一个优化问题。因此，不同的元启发式算法因其在合理计算时间内解决复杂优化问题的探索能力而被应用于双聚类问题。虽然已经提出了关于双聚类的各种调查，但缺乏使用超启发式方法对双聚类问题进行全面调查。本章将对解决双聚类问题的元启发式方法进行综述。本文综述了基本的优化方法及其主要搜索组件：表示、目标函数和变异算子。对单目标和多目标方法进行了具体讨论。最后，提出了一些新的研究方向。

<br><br>

【15】 A systematic review and meta-analysis of Digital Elevation Model (DEM) fusion: pre-processing, methods and applications<br>
**标题**：数字高程模型(DEM)融合的系统回顾和荟萃分析：前处理、方法和应用<br>
**链接**：https://arxiv.org/abs/2203.15026<br>

**作者**：Chukwuma Okolie,Julian Smit<br>
**机构**： School of Architecture Planning and Geomatics, University of Cape Town, South Africa 2Department of Surveying & Geoinformatics, University of Lagos<br>
**摘要**：遥感界已将数据融合确定为21世纪的关键挑战性课题之一。二维（2D）空间中的图像融合主题已在几篇发表的评论中涉及。然而，迄今为止，2.5D/3D数字高程模型（DEM）融合的特殊情况尚未得到解决。DEM融合是数据融合在遥感中的重要应用。它利用多源DEM的互补特性，提供更完整、准确和可靠的高程数据集。尽管已经开发出几种融合DEM的方法，但缺乏全面的审查限制了它们在研究人员和最终用户中的扩散。通常需要将多个研究的知识结合起来，以提供整体视角，并指导进一步的研究。作为回应，本文系统地回顾了DEM融合：预处理工作流程、方法和应用，并通过元分析进行了增强。通过讨论和比较分析，确定了尚未解决的挑战和未决问题，并提出了未来的研究方向。对于遥感和空间信息科学领域的研究人员以及整个数据融合社区来说，本综述是一个及时的解决方案，也是一个宝贵的信息来源。

<br><br>

【16】 Concept Embedding Analysis: A Review<br>
**标题**：概念嵌入分析研究述评<br>
**链接**：https://arxiv.org/abs/2203.13909<br>

<br><br>

【17】 Vision-and-Language Navigation: A Survey of Tasks, Methods, and Future Directions<br>
**标题**：视觉和语言导航：任务、方法和未来方向综述<br>
**链接**：https://arxiv.org/abs/2203.12667<br>

**作者**：Jing Gu,Eliana Stefani,Qi Wu,Jesse Thomason,Xin Eric Wang<br>
**机构**：University of California, Santa Cruz, The University of Adelaide, University of Southern California<br>
**摘要**：人工智能研究的一个长期目标是构建能够用自然语言与人类交流、感知环境和执行现实任务的智能代理。视觉和语言导航（VLN）是实现这一目标的基础性和跨学科研究课题，受到自然语言处理、计算机视觉、机器人和机器学习界越来越多的关注。在本文中，我们回顾了VLN新兴领域的当代研究，包括任务、评估指标、方法等。通过对当前进展和挑战的结构化分析，我们强调了当前VLN的局限性和未来工作的机会。本文为VLN研究界提供了全面的参考。

<br><br>

【18】 Community-Driven Comprehensive Scientific Paper Summarization: Insight from cvpaper.challenge<br>
**标题**：社区驱动的综合性科学论文综述：来自cvPaper的洞察力.Challenges<br>
**链接**：https://arxiv.org/abs/2203.09109<br>

**作者**：Shintaro Yamamoto,Hirokatsu Kataoka,Ryota Suzuki,Seitaro Shinagawa,Shigeo Morishima<br>
**摘要**：本文介绍了一项由志愿者参与者撰写会议记录摘要的小组活动。科学论文的快速增长对研究人员来说是一个沉重的负担，尤其是非母语人士，他们需要调查科学文献。为了缓解这个问题，我们组织了一个非英语母语人士小组，对计算机视觉会议上提交的论文进行总结，以分享该小组阅读的论文知识。我们总结了在2019年和2020年的顶级计算机视觉会议计算机视觉和模式识别会议上提交的总共2000篇论文。我们定量分析了参与者在众多可用论文中选择阅读哪些论文。实验结果表明，我们可以在不要求参与者阅读与其兴趣无关的论文的情况下，对大量论文进行总结。

<br><br>

【19】 A Systematic Review on Computer Vision-Based Parking Lot Management Applied on Public Datasets<br>
**标题**：基于公共数据集的基于计算机视觉的停车场管理系统综述<br>
**链接**：https://arxiv.org/abs/2203.06463<br>

**作者**：Paulo Ricardo Lisboa de Almeida,Jeovane Honório Alves,Rafael Stubs Parpinelli,Jean Paul Barddal<br>
**机构**：Department of Informatics, Federal University of Paran´a (UFPR), Curitiba (PR), Brazil, Graduate Program in Applied Computing, Santa Catarina State University (UDESC), Joinville (SC), Brazil<br>
**摘要**：基于计算机视觉的停车场管理方法因其灵活性和成本效益而受到广泛研究。为了评估这些方法，作者通常使用公开的停车场图像数据集。在这项研究中，我们调查和比较了可靠的公共可用图像数据集，这些数据集是专门为测试基于计算机视觉的停车场管理方法而设计的，因此，我们对使用这些数据集的现有作品进行了系统和全面的回顾。文献综述确定了需要进一步研究的相关差距，例如独立于数据集的方法和适合自动检测停车位位置的方法的要求。此外，我们注意到，在大多数研究中忽略了几个重要因素，例如连续图像中存在相同的车辆，从而导致评估协议不切实际。此外，对数据集的分析还表明，在制定新基准时应具备的某些特征，例如在更多样化的条件下拍摄的视频序列和图像的可用性，包括夜间和降雪，尚未纳入。

<br><br>

【20】 Deep Learning for Underwater Fish-Habitat Monitoring: A Survey<br>
**标题**：深度学习在水下鱼类栖息地监测中的应用综述<br>
**链接**：https://arxiv.org/abs/2203.06951<br>

**作者**：Alzayat Saleh,Marcus Sheaves,Mostafa Rahimi Azghadi<br>
**机构**：|, Cook University, Townsville, QLD, Australia, Correspondence, Science and Engineering, James Cook, Present address, †College of Science and Engineering, James, Funding information, This research is supported by an Australian<br>
**摘要**：海洋科学家利用远程水下视频记录来调查自然栖息地中的鱼类物种。这有助于他们理解和预测鱼类如何应对气候变化、栖息地退化和捕鱼压力。这些信息对于发展供人类食用的可持续渔业和保护环境至关重要。然而，收集到的大量视频使得提取有用信息对于人类来说是一项艰巨而耗时的任务。解决这个问题的一个很有希望的方法是尖端的深度学习（DL）技术。DL可以帮助海洋科学家快速有效地解析大量视频，解锁传统手动监控方法无法获得的生态位信息。在本文中，我们概述了DL的关键概念，同时对鱼类栖息地监测的文献进行了综述，重点介绍了水下鱼类分类。我们还讨论了为水下图像处理开发DL时面临的主要挑战，并提出了解决这些挑战的方法。最后，我们提供了对海洋栖息地监测研究领域的见解，并阐明了DL在水下图像处理方面的未来。本文旨在为广大读者提供信息，从希望在研究中应用DL的海洋科学家到希望调查基于DL的最新水下鱼类栖息地监测文献的计算机科学家。

<br><br>

【21】 Augmented Reality and Robotics: A Survey and Taxonomy for AR-enhanced Human-Robot Interaction and Robotic Interfaces<br>
**标题**：增强现实与机器人：增强现实增强的人-机器人交互和机器人接口的综述和分类<br>
**链接**：https://arxiv.org/abs/2203.03254<br>

**作者**：Ryo Suzuki,Adnan Karim,Tian Xia,Hooman Hedayati,Nicolai Marquardt<br>
**机构**：University of Calgary, Calgary, AB, Canada, UNC at Chapel Hill, Chapel Hill, NC, U.S.A., University College London, London, U.K., à, Reducing cognitive load, AR-based Safety Trade-offs, Technological challenges, Bridging gap between, studies and systems, In-the-wild deployments<br>
**摘要**：本文基于对460篇研究论文的调查，对增强现实技术和机器人技术进行了分类。增强和混合现实（AR/MR）已成为增强人机交互（HRI）和机器人接口（例如，驱动和形状变化接口）的一种新方法。最近，越来越多关于人机交互、HRI和机器人技术的研究表明，AR如何实现人与机器人之间更好的交互。然而，研究通常集中在个人探索和关键设计策略上，很少系统地分析研究问题。在本文中，我们从以下几个方面对这一研究领域进行了综合和分类：1）增强现实的方法；2） 机器人的特点；3） 目的和利益；4） 呈现信息的分类；5） 设计视觉增强的组件和策略；6） 互动技巧和方式；7） 应用领域；8）评估策略。我们制定了关键挑战和机遇，以指导和告知AR和机器人技术的未来研究。

<br><br>

【22】 Recent Advances and Challenges in Deep Audio-Visual Correlation Learning<br>
**标题**：视听深度相关学习的最新进展与挑战<br>
**链接**：https://arxiv.org/abs/2202.13673<br>

**作者**：Luís Vilaça,Yi Yu,Paula Viana<br>
**摘要**：视听相关学习旨在捕捉音频和视频之间的基本对应关系，理解自然现象。随着深度学习的迅速发展，这一新兴的研究课题越来越受到重视。在过去的几年里，各种方法和数据集被提出用于视听相关学习，这促使我们进行全面的调查。本文重点介绍了用于学习音频和视频之间相关性的最新模型（SOTA），同时也讨论了人工智能多媒体中的一些定义和范例任务。此外，我们还研究了一些常用于优化视听相关学习模型的目标函数，并讨论了在优化过程中如何利用视听数据。最重要的是，我们对SOTA视听相关学习的最新进展进行了广泛的比较和总结，并讨论了未来的研究方向。

<br><br>

【23】 A Survey of Vision-Language Pre-Trained Models<br>
**标题**：视觉语言预训练模型综述<br>
**链接**：https://arxiv.org/abs/2202.10936<br>

**作者**：Yifan Du,Zikang Liu,Junyi Li,Wayne Xin Zhao<br>
**机构**：Gaoling School of Artificial Intelligence, Renmin University of China, DIRO, Universit´e de Montr´eal<br>
**摘要**：随着Transformer的发展，经过预训练的模型近年来以惊人的速度发展。它们主导了自然语言处理（NLP）和计算机视觉（CV）的主流技术。如何使预训练适应视野和语言（V-L）学习，并提高下游任务的绩效成为多模式学习的一个焦点。本文综述了视觉语言预训练模型（VL-PTM）的最新进展。作为核心内容，我们首先简要介绍了在预训练之前将原始图像和文本编码为单模态嵌入的几种方法。然后，我们深入研究了VL PTMs的主流体系结构，对文本和图像表示之间的交互进行建模。我们进一步介绍了广泛使用的训练前任务，然后介绍了一些常见的下游任务。最后，我们对本文进行了总结，并提出了一些有前景的研究方向。我们的调查旨在为多模态研究人员提供相关研究的综合和指针。

<br><br>

【24】 A Review of Emerging Research Directions in Abstract Visual Reasoning<br>
**标题**：抽象视觉推理新兴研究方向述评<br>
**链接**：https://arxiv.org/abs/2202.10284<br>

**作者**：Mikołaj Małkiński,Jacek Mańdziuk<br>
**摘要**：抽象视觉推理（AVR）问题通常用于近似人类智能。他们测试在一个全新的环境中应用之前获得的知识、经验和技能的能力，这使他们特别适合这项任务。最近，AVR问题已经成为研究机器智能的一个热门代理，这导致了新的不同类型的问题和多个基准集的出现。在这项工作中，我们回顾了这项新兴的AVR研究，并提出了一种分类法，将AVR任务分为5个维度：输入形状、隐藏规则、目标任务、认知功能和主要挑战。本次调查所采用的视角可以描述AVR问题的共同和独特性质，提供解决AVR任务的现有方法的统一观点，展示AVR问题与实际应用的关系，并概述未来工作的前景。其中之一是机器学习文献中的观察结果，即不同的任务被孤立地考虑，这与AVR任务用于测量人类智力的方式形成了鲜明对比，在AVR任务中，多种类型的问题被结合在一个IQ测试中。

<br><br>

【25】 A Comprehensive Survey with Quantitative Comparison of Image Analysis Methods for Microorganism Biovolume Measurements<br>
**标题**：微生物生物体积测量图像分析方法的综述与定量比较<br>
**链接**：https://arxiv.org/abs/2202.09020<br>

**作者**：Jiawei Zhang,Chen Li,Md Mamunur Rahaman,Yudong Yao,Pingli Ma,Jinghua Zhang,Xin Zhao,Tao Jiang,Marcin Grzegorzek<br>
**摘要**：随着城市化进程的加快和人们生活水平的提高，微生物在工业生产、生物技术和食品安全检测中发挥着越来越重要的作用。微生物生物量测量是微生物分析的重要组成部分之一。然而，传统的手工测量方法费时费力，难以精确测量其特性。随着数字图像处理技术的发展，微生物种群的特征可以被检测和量化。可以及时调整变化趋势，为改进提供依据。自20世纪80年代以来，微生物生物体积测量方法的应用得到了发展。本研究综述了60多篇文章，并采用带周期的数字图像分割方法对这些文章进行分组。本研究具有很高的研究意义和应用价值，可供微生物研究人员利用数字图像分析方法全面了解微生物生物体积测量及其潜在应用。

<br><br>

【26】 Avoiding Overfitting: A Survey on Regularization Methods for Convolutional Neural Networks<br>
**标题**：避免过拟合：卷积神经网络正则化方法综述<br>
**链接**：https://arxiv.org/abs/2201.03299<br>

**作者**：Claudio Filipi Gonçalves dos Santos,João Paulo Papa<br>
**机构**：and Eldorado’s Institute of Technology, Brazil<br>
**摘要**：一些图像处理任务，如图像分类和目标检测，已通过卷积神经网络（CNN）得到显著改进。与ResNet和EfficientNet一样，许多体系结构在创建时至少在一个数据集上取得了优异的成绩。训练中的一个关键因素是网络的正则化，它可以防止结构过度拟合。这项工作分析了过去几年中发展的几种正则化方法，显示了不同CNN模型的显著改进。这些工作分为三个主要领域：第一个领域称为“数据扩充”，所有技术都集中于对输入数据进行更改。第二种称为“内部变化”，旨在描述修改由神经网络或内核生成的特征映射的过程。最后一个称为“标签”，涉及转换给定输入的标签。与其他关于规范化的可用调查相比，这项工作呈现出两个主要差异：（i）第一个与手稿中收集的论文有关，不超过五年；（ii）第二个区别与再现性有关，即。，这里提到的所有作品都在公共存储库中提供了它们的代码，或者它们直接在一些框架中实现，比如TensorFlow或Torch。

<br><br>

【27】 The State of Aerial Surveillance: A Survey<br>
**标题**：空中侦察现状综述<br>
**链接**：https://arxiv.org/abs/2201.03080<br>

**作者**：Kien Nguyen,Clinton Fookes,Sridha Sridharan,Yingli Tian,Xiaoming Liu,Feng Liu,Arun Ross<br>
**机构**：•, Science and Engineering, Michigan State University, United States. E-, satellites and is sufficient to detect a mobile phone in a, person’s hand and track all individual vehicles day and, night in an entire city for weeks, even months [,].<br>
**摘要**：由于机载平台和成像传感器在规模、机动性、部署和隐蔽观测能力方面具有前所未有的优势，它们的迅速出现使得新形式的空中监视成为可能。本文从计算机视觉和模式识别的角度全面概述了以人为中心的空中监视任务。它的目的是为读者提供一个深入系统的审查和使用无人机，无人机和其他机载平台的空中监视任务的当前状态的技术分析。感兴趣的主要对象是人类，其中单个或多个主体将被检测、识别、跟踪、重新识别，并对其行为进行分析。更具体地说，对于这四项任务中的每一项，我们首先讨论在空中环境中执行这些任务与地面环境相比所面临的独特挑战。然后，我们回顾和分析每项任务公开的航空数据集，深入研究航空文献中的方法，并调查它们目前如何应对航空挑战。我们在结束论文时讨论了缺失的差距和开放的研究问题，以告知未来的研究途径。

<br><br>

【28】 Data Harmonisation for Information Fusion in Digital Healthcare: A State-of-the-Art Systematic Review, Meta-Analysis and Future Research Directions<br>
**标题**：数字医疗信息融合中的数据协调：最新系统回顾、荟萃分析和未来研究方向<br>
**链接**：https://arxiv.org/abs/2201.06505<br>

**作者**：Yang Nan,Javier Del Ser,Simon Walsh,Carola Schönlieb,Michael Roberts,Ian Selby,Kit Howard,John Owen,Jon Neville,Julien Guiot,Benoit Ernst,Ana Pastor,Angel Alberich-Bayarri,Marion I. Menzel,Sean Walsh,Wim Vos,Nina Flerin,Jean-Paul Charbonnier,Eva van Rikxoort,Avishek Chatterjee,Henry Woodruff,Philippe Lambin,Leonor Cerdá-Alberich,Luis Martí-Bonmatí,Francisco Herrera,Guang Yang<br>
**机构**： National Heart and Lung Institute, Imperial College London, London, UK, Cardiovascular Research Centre, Royal Brompton Hospital, London, UK, School of Biomedical Engineering & Imaging Sciences, King's College London, London, UK<br>

**摘要**：消除多中心数据的偏差和方差一直是大规模数字医疗研究中的一个挑战，这要求能够集成从不同扫描仪和协议获取的数据中提取的临床特征，以提高稳定性和鲁棒性。先前的研究描述了融合单模态多中心数据集的各种计算方法。然而，这些调查很少关注评估指标，也缺乏计算数据协调研究的检查表。在本系统综述中，我们总结了数字医疗领域中多模态数据的计算数据协调方法，包括基于不同理论的协调策略和评估指标。此外，还提出了一份全面的清单，总结了数据协调研究的常见做法，以指导研究人员更有效地报告其研究结果。最后，提出了方法和指标选择的可能方法流程图，并对不同方法的局限性进行了调查，以供未来研究。

<br><br>

【29】 A Survey on RGB-D Datasets<br>
**标题**：关于RGB-D数据集的综述<br>
**链接**：https://arxiv.org/abs/2201.05761<br>

**作者**：Alexandre Lopes,Roberto Souza,Helio Pedrini<br>
**机构**：Institute of Computing, University of Campinas, Brazil, Department of Electrical and Software Engineering, University of Calgary, Canada<br>
**摘要**：RGB-D数据对于解决计算机视觉中的许多问题至关重要。已经提出了数百个包含各种场景的公共RGB-D数据集，例如室内、室外、空中、驾驶和医疗场景。这些数据集适用于不同的应用，是解决经典计算机视觉任务（如单目深度估计）的基础。本文回顾并分类了包含深度信息的图像数据集。我们收集了203个包含可访问数据的数据集，并将它们分为三类：场景/对象、身体和医疗。我们还概述了不同类型的传感器、深度应用，并研究了包含深度数据的数据集的使用和创建的趋势和未来方向，以及如何将其应用于研究单目深度估计领域中可推广机器学习模型的发展。

<br><br>

【30】 Towards deep observation: A systematic survey on artificial intelligence techniques to monitor fetus via Ultrasound Images<br>
**标题**：走向深度观察：人工智能超声图像胎儿监护技术的系统研究<br>
**链接**：https://arxiv.org/abs/2201.07935<br>

**作者**：Mahmood Alzubaidi,Marco Agus,Khalid Alyafei,Khaled A Althelaya,Uzair Shah,Alaa A. Abdalrazaq,Mohammed Anbar,Zafar Iqbal,Mowafa Househ<br>
**机构**： College of Science and Engineering, Hamad Bin Khalifa University, Doha, Qatar; , Weil Cornell Medical College-Qatar; ,Sidra, Medical and Research Center, Doha, Qatar; , National Advanced IPv, Centre, University Sains Malaysia, Penang, Malaysia<br>
**摘要**：发展创新的信息学方法以加强胎儿监护是生殖医学的一个新兴研究领域。关于人工智能（AI）技术改善妊娠结局已经进行了几次审查。他们的局限性在于关注特定的数据，如怀孕期间母亲的护理。本系统调查旨在探索人工智能（AI）如何通过超声（US）图像辅助胎儿生长监测。我们使用了八个医学和计算机科学书目数据库，包括PubMed、Embase、PsycINFO、ScienceDirect、IEEE explore、ACM图书馆、Google Scholar和科学网。我们检索了2010至2021年间发表的研究结果。从研究中提取的数据采用叙述法进行综合。在1269项检索到的研究中，我们包括107项与调查主题相关的不同研究。我们发现2D超声图像比3D和4D超声图像（n=19）更受欢迎（n=88）。分类是最常用的方法（n=42），其次是分割（n=31）、与分割相结合的分类（n=16）和其他杂项方法，如目标检测、回归和强化学习（n=18）。妊娠区域内最常见的区域是胎儿头部（n=43），然后是胎儿身体（n=31），胎儿心脏（n=13），胎儿腹部（n=10），最后是胎儿面部（n=10）。在最近的研究中，深度学习技术主要使用（n=81），其次是机器学习（n=16）、人工神经网络（n=7）和强化学习（n=2）。人工智能技术在预测胎儿疾病和识别妊娠期胎儿解剖结构方面发挥了关键作用。需要更多的研究从医生的角度来验证这项技术，例如人工智能及其在医院环境中的应用的试点研究和随机对照试验。

<br><br>

【31】 A Comprehensive Survey on Federated Learning: Concept and Applications<br>
**标题**：联邦学习：概念及其应用综述<br>
**链接**：https://arxiv.org/abs/2201.09384<br>

**作者**：Dhurgham Hassan Mahlool,Mohammed Hamzah Abed<br>
**机构**：AL-Qadisiyah University, College of Computer Science and Information, Technology , Computer Science Department, Corresponding Author<br>
**摘要**：本文对联邦学习（FL）进行了全面的研究，重点介绍了其组成、面临的挑战、应用和FL环境。FL可以应用于现实模型中的多个领域。在医疗系统中，患者记录及其医疗状况的隐私是关键数据，因此协作学习或联合学习应运而生。另一方面，构建一个智能系统来帮助医务人员，而无需共享FL概念中的数据，使用的应用之一是基于AI方法的脑肿瘤诊断智能系统，该系统可以在协作环境中高效工作。本文将介绍一些在医学领域的应用和相关工作，以及FL概念下的工作，然后总结它们，介绍其工作的主要局限性。

<br><br>

【32】 Spectral, Probabilistic, and Deep Metric Learning: Tutorial and Survey<br>
**标题**：光谱、概率和深度度量学习：教程和综述<br>
**链接**：https://arxiv.org/abs/2201.09267<br>

**作者**：Benyamin Ghojogh,Ali Ghodsi,Fakhri Karray,Mark Crowley<br>
**机构**：Department of Electrical and Computer Engineering, Machine Learning Laboratory, University of Waterloo, Waterloo, ON, Canada, Department of Statistics and Actuarial Science & David R. Cheriton School of Computer Science<br>
**摘要**：这是一篇关于度量学习的教程和调查论文。算法分为谱学习、概率学习和深度度量学习。我们首先从距离度量、马氏距离和广义马氏距离的定义开始。在光谱方法中，我们从使用分散数据的方法开始，包括第一个光谱度量学习、Fisher判别分析的相关方法、相关成分分析（RCA）、判别成分分析（DCA）和Fisher HSIC方法。然后介绍了大范围度量学习、非平衡度量学习、局部线性度量自适应和对抗性度量学习。我们还解释了几种用于特征空间中度量学习的核谱方法。我们还介绍了黎曼流形上的几何度量学习方法。在概率方法中，我们从在输入空间和特征空间中折叠类开始，然后解释邻域成分分析方法、贝叶斯度量学习、信息论方法以及度量学习中的经验风险最小化。在深度学习方法中，我们首先介绍了重建自动编码器和用于度量学习的监督损失函数。然后，解释了暹罗网络及其各种损失函数、三重态挖掘和三重态采样。本文还对基于Fisher判别分析的深度判别分析方法进行了综述。最后，我们介绍了多模态深度度量学习、神经网络几何度量学习和Few-Shot度量学习。

<br><br>

【33】 Learning-Driven Lossy Image Compression; A Comprehensive Survey<br>
**标题**：学习驱动的有损图像压缩研究综述<br>
**链接**：https://arxiv.org/abs/2201.09240<br>

**作者**：Sonain Jamil,Md. Jalil Piran,MuhibUrRahman<br>
**机构**： SejongUniversity<br>
**摘要**：在图像处理和计算机视觉领域，机器学习（ML）体系结构得到了广泛的应用。卷积神经网络（CNN）解决了广泛的图像处理问题，可以解决图像压缩问题。由于带宽和内存限制，图像压缩是必要的。有用信息、冗余信息和无关信息是在图像中发现的三种不同形式的信息。本文旨在综述使用ML体系结构的主要有损图像压缩的最新技术，包括不同的自动编码器（AEs），如卷积自动编码器（CAE）、变分自动编码器（VAE）和具有超先验模型的AEs、递归神经网络（RNN）、CNN、生成性对抗网络（GAN），主成分分析（PCA）和模糊均值聚类。我们根据架构将所有算法分为几个组。我们在这项调查中介绍了静态图像压缩。强调了研究人员的各种发现，以及研究人员未来可能的方向。解释了内存不足（OOM）、条带区域失真（SRD）、混叠以及框架与中央处理器（CPU）和图形处理器（GPU）的兼容性等开放性研究问题。所调查的压缩领域的大多数出版物都来自前五年，使用了多种方法。

<br><br>

【34】 A Survey on Patients Privacy Protection with Stganography and Visual Encryption<br>
**标题**：基于数字图像和视觉加密的患者隐私保护调查<br>
**链接**：https://arxiv.org/abs/2201.09388<br>

**作者**：Hussein K. Alzubaidy,Dhiah Al-Shammary,Mohammed Hamzah Abed
**机构**：College of Computer Science and Information TechnologyUniversity of Al-, Qadisiyah Dewaniyah Iraq, Corresponding Author<br>
**摘要**：在这项调查中，讨论了30种隐写术和视觉加密方法模型，以保护患者隐私。

<br><br>

【35】 Deep Learning Methods for Abstract Visual Reasoning: A Survey on Raven's Progressive Matrices<br>
**标题**：抽象视觉推理的深度学习方法--瑞文递进矩阵研究综述<br>
**链接**：https://arxiv.org/abs/2201.12382<br>

**作者**：Mikołaj Małkiński,Jacek Mańdziuk<br>
**摘要**：抽象视觉推理（AVR）领域包含解决问题的能力，需要对给定场景中存在的实体之间的关系进行推理。虽然人类通常会以“自然”的方式解决AVR任务，即使之前没有经验，但事实证明，对于当前的机器学习系统来说，这类问题很难解决。本文总结了应用深度学习方法解决AVR问题的最新进展，作为研究机器智能的替代。我们关注最常见的AVR任务类型——Raven的渐进矩阵（RPM）——并全面回顾了用于求解RPM的学习方法和深度神经模型，以及RPM基准集。通过对解决RPM的最先进方法的性能分析，可以对该领域的当前和未来趋势提出一些见解和意见。我们通过展示现实世界的问题如何从RPM研究的发现中受益来结束本文。

<br><br>

【36】 Performance Evaluation of Infrared Image Enhancement Techniques<br>
**标题**：红外图像增强技术的性能评价<br>
**链接**：https://arxiv.org/abs/2202.03427<br>

**作者**：Rania Gaber,AbdElmgied Ali,Kareem Ahmed<br>
**摘要**：红外（IR）图像在医学成像、目标跟踪、天文学和军事等领域有着广泛的应用。根据拍摄设备的类型，可以白天或晚上拍摄红外图像。捕获设备使用波长更长的电磁辐射。根据波长范围和相应频率，有几种类型的红外辐射。由于噪声和其他伪影，红外图像不清晰可见。本文对红外成像增强技术进行了全面的综述。调查包括红外辐射类型和设备以及现有的红外数据集。该调查涵盖了空间增强技术、基于频域的增强技术和基于深度学习的技术。

<br><br>

【37】 A Survey of Neural Trojan Attacks and Defenses in Deep Learning<br>
**标题**：深度学习中的神经木马攻击与防御综述<br>
**链接**：https://arxiv.org/abs/2202.07183<br>

**作者**：Jie Wang,Ghulam Mubashar Hassan,Naveed Akhtar<br>
**摘要**：人工智能（AI）在很大程度上依赖于深度学习——这项技术在人工智能的实际应用中越来越流行，甚至在安全关键和高风险领域也是如此。然而，最近发现，深度学习可以通过在其中嵌入特洛伊木马进行操作。不幸的是，规避深度学习计算要求的实用解决方案（例如，将模型训练或数据注释外包给第三方）进一步增加了模型对特洛伊木马攻击的敏感性。由于该主题在深度学习中的关键重要性，最近的文献在这方面做出了许多贡献。我们对设计特洛伊木马攻击以进行深入学习的技术进行了全面回顾，并探讨了它们的防御措施。我们的信息调查系统地组织了最近的文献，并讨论了方法的关键概念，同时假设读者对该领域的知识最少。它为更广泛的社区了解神经特洛伊木马的最新发展提供了一个可理解的途径。

<br><br>

【38】 VLP: A Survey on Vision-Language Pre-training<br>
**标题**：VLP：关于视觉语言前期训练的调查<br>
**链接**：https://arxiv.org/abs/2202.09061<br>

**作者**：Feilong Chen,Duzhan Zhang,Minglun Han,Xiuyi Chen,Jing Shi,Shuang Xu,Bo Xu<br>
**摘要**：在过去几年中，训练前模型的出现将计算机视觉（CV）和自然语言处理（NLP）等单峰领域带入了一个新时代。大量工作表明，它们有利于下游单峰任务，避免从头开始训练新模型。那么，这种预先训练好的模型能应用于多模态任务吗？研究人员已经探索了这个问题，并取得了重大进展。本文综述了视觉语言预训练（VLP）的最新进展和新前沿，包括图像文本和视频文本预训练。为了让读者更好地全面了解VLP，我们首先从五个方面回顾了VLP的最新进展：特征提取、模型体系结构、预训练目标、预训练数据集和下游任务。然后，我们详细总结了具体的VLP模型。最后，我们讨论了VLP的新前沿。据我们所知，这是第一次关于VLP的调查。我们希望这项调查能够为VLP领域的未来研究提供一些启示。

<br><br>

【39】 ScePT: Scene-consistent, Policy-based Trajectory Predictions for Planning<br>
**标题**：SCEPT：场景一致、基于策略的规划轨迹预测<br>
**链接**：https://arxiv.org/abs/2206.13387<br>

**作者**：Yuxiao Chen,Boris Ivanovic,Marco Pavone<br>
**摘要**：轨迹预测是与非受控主体共享环境的自治系统的一项关键功能，自动驾驶车辆就是一个突出的例子。目前，大多数预测方法并不强制场景一致性，即场景中不同代理的预测轨迹之间存在大量的自碰撞。此外，许多方法生成每个agent的单独轨迹预测，而不是整个场景的联合轨迹预测，这使得下游规划变得困难。在这项工作中，我们提出了ScePT，这是一种基于策略规划的轨迹预测模型，可以生成精确的、场景一致的轨迹预测，适用于自主系统运动规划。它明确地强制场景一致性，并学习可用于条件预测的代理交互策略。在多个真实行人和自主车辆数据集上的实验表明，ScePT}与当前最先进的预测精度相匹配，场景一致性显著提高。我们还展示了ScePT与下游应急计划员合作的能力。

<br><br>

【40】 All One Needs to Know about Priors for Deep Image Restoration and Enhancement: A Survey<br>
**标题**：深度图像恢复和增强的先验知识：综述<br>
**链接**：https://arxiv.org/abs/2206.02070<br>

**作者**：Yunfan Lu,Yiqi Lin,Hao Wu,Yunhao Luo,Xu Zheng,Lin Wang<br>
**摘要**：图像恢复和增强是通过消除噪声、模糊和分辨率下降等退化来提高图像质量的过程。深度学习（DL）最近被应用于图像恢复和增强。由于其不适定特性，大量工作探索了先验知识，以便于训练深层神经网络（DNN）。然而，到目前为止，研究界尚未对先验知识的重要性进行系统的研究和分析。因此，本文首次全面概述了先验知识在深度图像恢复和增强方面的最新进展。我们的工作包括五个主要内容：（1）对深度图像恢复和增强的先验知识进行了理论分析；（2） 基于DL的方法中常用的优先级的层次和结构分类法；（3） 深入讨论每个先验知识的原理、潜力和应用；（4） 通过突出潜在的未来方向，总结关键问题，以激发社区内更多的研究；（5） 一个开放源代码存储库，提供所有提到的作品和代码链接的分类。
