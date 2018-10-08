
## Kache.ai - Annotation Guidelines

The goal of this post is to provide instructions for the new machine labeling policy. These guidelines also serve as a getting started guide to introduce the [Scalabel Platform](http://scalabel.ai), created by the [Berkeley Deep Drive Team](http://bdd-data.berkeley.edu/).


### Scalabel Labeling Tool

Migrating to this tool provides us with a variety of advantages over the legacy systems (LabelMe, VGG, etc). For example, scalabel allow us to tag images with custom attributes. In addition to labeling categories (e.g., cars), we can advantageously encode features from the environment such as the **occlusion** and **truncation** parameters.
In addition, Scalabel provides a flexible and robust data format that supports labeling pipelines particularly suited for autonomous driving. We will be utilizing Scalabel as the central tool for providing human annotations moving forward.


### Getting Started - Navigating the Web App

To create a project, navigate to the [Scalabel app](http://ec2-52-25-35-71.us-west-2.compute.amazonaws.com:8686/) and simply upload the relevant **annotations file** for correction. As an example, I preloaded the BDD100k dataset for visualization:

---

|  BDD100K - PT1    |  BDD100K - PT2   |  BDD100K - PT3    |  BDD100K - PT4    |
|:-----------------:|:----------------:|:-----------------:|:-----------------:|
|    [PT1](http://ec2-52-25-35-71.us-west-2.compute.amazonaws.com:8686/label2d?project_name=BDD100K-Kache-PT1_4&task_index=0)   |   [PT2](http://ec2-52-25-35-71.us-west-2.compute.amazonaws.com:8686/label2d?project_name=BDD100K-Kache-PT2_4&task_index=0)   |   [PT3](http://ec2-52-25-35-71.us-west-2.compute.amazonaws.com:8686/label2d?project_name=BDD100K-Kache-PT3_4&task_index=0)   |    [PT4](http://ec2-52-25-35-71.us-west-2.compute.amazonaws.com:8686/label2d?project_name=BDD100K-Kache-PT4_4&task_index=0)   |

---

### Getting Started - Creating a New Project


- Once you see the screen shown below, click **Create A New Project**
- Next, choose a project name (No spaces allowed). For bounding box corrections, click on **Image** and **2D Bounding Box** from the dropdown menus. Depending on the project, the next uploads will vary depending on the task, but the scalabel examples provide the **categories.yml** and **bbox_attributes.yml** configuration files.
- Upload your **annotations file** into the Item list (*see below*).
- Select a **Task Size** this size determines the number of chunks you want to break up your annotations file. For example, if you have an item list of **10,000** images and your task size is 10, the you will have generated **10 task lists**; each with **1000** items. The 10 task lists will be assigned to the project name of your choosing.
- The vendor id will be **0** for now, in the future we will use this field to represent different users of a company (for QA purposes).


|   Project Creation - Scalabel   |
|:-------------------------------:|
|      ![homepage][image1]       |
|     ![create][image2]          |
|     ![upload_cfgs][image3]     |

### Uploading an Annotations File

Generally, this file will be provided by internal scripts which select images we intend to use for training nets. Upload the given document into the *item list*.

---

![upload_anns][image4]

---

And that's it. Clicking **Enter** will generate a new task list.

---

![post_proj][image5]

---

Click on **Go To Project Dashboard**, the next page will give you the option to click one the tasks generated and begin annotating. Also from the dashboard, you will have the option to <font color='red'>export your results into bdd format for training</font>

---

![proj_dash][image6]

---


---

![proj_begin][image7]

---

## Style Guides:

In this section, we will explore the BDD distibution and define our policy for labeling.

---

### Style Guides: <font color='green'>Good Techniques</font>

We would like to continue to maintain BDD's granularity and attribute associations in all new images. this will allow us to easily blend our interal data with the BDD distribution. In particular, we will maintain using occlusion and truncation for all pertaining objects. For our purposes, the respective attributes are defined as follows:
 - **Occlusion:** When one object is hidden by another object that passes between it and the observer. The term refers to any situation in which an object in the foreground blocks from view (occults) an object in the background. In our sense, occlusion applies to the visual scene observed from computer-generated imagery when foreground objects obscure distant objects dynamically, as the scene changes over time.

 - **Truncation:** The bounding box of the object specified does not correspond to the full extent of the object e.g. an image of a person from the waist up, or a view of a car extending outside the image.

---

### Style Guides: <font color='red'>Mis-Classifications / Class-Ambiguities</font>

Overall the BDD dataset is remarkably accurate and the class ambiguities are nuanced, however there are some consistent miscategorizations that we would like fixed in our final production dataset. Among those subtleties, a few features we should aim to correct are as follows:

---

### Truck Mis-Labels
---

|   Trucks Class Ambiguities      |
|:-------------------------------:|
|     ![truck_v_car1][image8]     |
|     ![truck_v_car2][image9]     |
|     ![truck_v_car3][image10]    |

---


Our definition is somewhat different from the definition in the BDD dataset. We limit the trucks class to vehicles which **require a class A license or similar to drive** examples of improper trucks (e.g., pickups) classes are shown above. These vehicles must be changed to car class.

Another **proper** truck type are the [**Towing Wrecker Vehicles**](https://www.google.com/search?q=towing+wreckers&source=lnms&tbm=isch&sa=X&ved=0ahUKEwiq7JCEsvfdAhWFJnwKHRfNB-gQ_AUIECgD&biw=1855&bih=990). The example above is a tow truck but should be labeled as a car since its towing capacity is similar to a modified pickup. We consider this type of truck in a different class than the larger breed shown in the hyperlink above.


### Bus Mis-Labels

---

![bus_v_car][image11]

---

Similar to the trucks class, our definition of a bus is more constrained. We will define a bus as only those vehicles which are large enough to have the typical bus-styled swinging door openening mechanism which attached to a turn crank. An example of a truck mis-label is shown above. In this case, we want to classify this vehicle as a bus because it likely has a door that need to swing out to open. As it is a shuttle, it is also likely to make frequent stops.

### Car Mis-Labels

---

|  Incorrect/Missing Car Labels
|:-------------------------------:|
|     ![car_antenna][image12]     |
|     ![car_mislabel0][image15]   |
|     ![car_mislabel1][image13]   |
|     ![car_mislabel2][image14]   |

---

With the exceptions to the above, our definition of cars is identical to BDD.. One small difference to note is that we will define the car bounding box to include the car itself, excluding protruding items like the car atenna mounted above the car.

In the other examples, the car classes are either a) mis-labeled altogether, or b) labeled, but missing the proper **occlusion** or **truncation** attrubutes. We must clean these instances up.

## Style Guides: Class Taxonomy

Another great feature of Scalabel is that you can easily change the classification template to whatever task suits your need. We intend to leverage these capabilities by customizing our class naming scheme. This guide will serve as the official instructions on which classes to label unless there are special circumstances.  As of now, those classes are:

- truck
- bus
- car
- traffic light (red, yellow(amber), green, unknown
- rider (person atop a motorized vehicle or bicycle)
- bike
- train
- motor
- person (pedestrians)
- traffic sign
- construction barrel
- construction post
- construction cone
- construction sign
- construction barrier

And many more soon to come.

---

### For Developers:

Here is the Json format which is supported by the training pipeline scripts currently in iPython notebooks *See e.g.*:
- **[`darkernet.py`](https://github.com/deanwebb/data_to_coco/blob/master/darknet_train_bdd.ipynb)**,(python wrapper around darknet.py, coming soon)
- **`rails-reactor API`**,
- **`bdd-formatter ros/atm modules`,** (coming soon)
- **[`data_to_coco`](https://github.com/deanwebb/data_to_coco)** scripts repo


---

[//]: # (Image References)
[image1]: readme_imgs/homepg.png
[image2]: readme_imgs/create_projs.png
[image3]: readme_imgs/upload_cfgs.png
[image4]: readme_imgs/upload_anns.png
[image5]: readme_imgs/post_proj.png
[image6]: readme_imgs/proj_dash.png
[image7]: readme_imgs/export_task_urls.png
[image8]: readme_imgs/truck_v_car1.png
[image9]: readme_imgs/truck_v_car2.png
[image10]: readme_imgs/truck_v_car3.png
[image11]: readme_imgs/bus_v_car.png
[image12]: readme_imgs/car_antenna.png
[image13]: readme_imgs/car_mislabel1.png
[image14]: readme_imgs/car_mislabel2.png
[image15]: readme_imgs/car_mislabel0.png
