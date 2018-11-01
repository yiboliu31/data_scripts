

# Kache.ai - Annotation Guidelines

The goal of this post is to provide instructions for the new machine labeling policy. These guidelines also serve as a getting started guide to introduce the [Scalabel Platform](http://scalabel.ai), created by the [Berkeley Deep Drive Team](http://bdd-data.berkeley.edu/).


## Scalabel Labeling Tool

Migrating to this tool provides us with a variety of advantages over the legacy systems (LabelMe, VGG, etc). For example, scalabel allow us to tag images with custom attributes. In addition to labeling categories (e.g., cars), we can advantageously encode features from the environment such as the **occlusion** and **truncation** parameters. 
In addition, Scalabel provides a flexible and robust data format that supports labeling pipelines particularly suited for autonomous driving. We will be utilizing Scalabel as the central tool for providing human annotations moving forward.

---

### Getting Started - Navigating the Web App

To create a project, navigate to the [Scalabel app](http://ec2-52-25-35-71.us-west-2.compute.amazonaws.com:8686/) and simply upload the relevant **annotations file** for correction. As an example, I preloaded the BDD100k dataset for visualization:

---

|  BDD100K - PT1    |  BDD100K - PT2   |  BDD100K - PT3    |  BDD100K - PT4    |
|:-----------------:|:----------------:|:-----------------:|:-----------------:|
|    [PT1](http://ec2-52-25-35-71.us-west-2.compute.amazonaws.com:8686/label2d?project_name=BDD100K-Kache-PT1_4&task_index=0)   |   [PT2](http://ec2-52-25-35-71.us-west-2.compute.amazonaws.com:8686/label2d?project_name=BDD100K-Kache-PT2_4&task_index=0)   |   [PT3](http://ec2-52-25-35-71.us-west-2.compute.amazonaws.com:8686/label2d?project_name=BDD100K-Kache-PT3_4&task_index=0)   |    [PT4](http://ec2-52-25-35-71.us-west-2.compute.amazonaws.com:8686/label2d?project_name=BDD100K-Kache-PT4_4&task_index=0)   |

---

### Getting Started - Creating a New Project & Uploading an Annotations File


* Once you see the screen shown below, click **Create A New Project** 
* Next, choose a project name (No spaces allowed). For bounding box corrections, click on **Image** and **2D Bounding Box** from the dropdown menus. Depending on the project, the next uploads will vary depending on the task, but the scalabel examples provide the **categories.yml** and **bbox_attributes.yml** configuration files.
* Upload your **annotations file** into the Item list (*see below*).
* Select a **Task Size** this size determines the number of chunks you want to break up your annotations file. For example, if you have an item list of **10,000** images and your task size is 10, the you will have generated **10 task lists**; each with **1000** items. The 10 task lists will be assigned to the project name of your choosing.
* The vendor id will be **0** for now, in the future we will use this field to represent different users of a company (for QA purposes). 


|   Project Creation - Scalabel   |

|  Homepage   |  Create Project |  Upload Configs | Upload Annotations |
|:-----------------:|:----------------:|:-----------------:|:-----------------:|
|  ![homepage][image1] | ![create][image2] | ![upload_cfgs][image3]|  ![upload_anns][image4] |



Generally, the annotations file will be provided by internal scripts or databse which selects images intended for training nets. Upload the given document into the *item list*.


---

And that's it. Clicking **Enter** will generate a new task list. Clicking on **Go To Project Dashboard** on the next page will give you the option to click one the tasks generated and begin annotating. Also from the dashboard, you will have the option to <font color='red'>export your results into bdd format for training</font> 


![proj_dash][image6]

---


![proj_begin][image7]

---

## Style Guide: 

In this section, we will explore the BDD distibution and define our policy for labeling. 

---

### Style Guides: <font color='green'>Good Techniques</font> 

We would like to continue to maintain BDD's granularity and attribute associations in all new images. this will allow us to easily blend our interal data with the BDD distribution. In particular, we will maintain using occlusion and truncation for all pertaining objects. For our purposes, the respective attributes are defined as follows:

- **Occlusion:** When one object is hidden by another object that passes between it and the observer. The term refers to any situation in which an object in the foreground blocks from view (occults) an object in the background. In our sense, occlusion applies to the visual scene observed from computer-generated imagery when foreground objects obscure distant objects dynamically, as the scene changes over time. 


| Occlusion Examples|
|:-----------------:|
|![][image68]|
|![][image69]|
|![][image70]|
|![][image71]|
 
 
- **Truncation:** The bounding box of the object specified does not correspond to the full extent of the object e.g. an image of a person from the waist up, or a view of a car extending outside the outside the field of view of the camera/image, such that it is partially shown. It is assumed with truncated images that a subsequent perspective shift will then clarify the bounds in the view of the truncated object.

|  Truncation Examples |
|:-----------------:|
| ![][image72]|
|![][image73]|
|![][image74]|
|![][image75] |

---

### Style Guides: <font color='red'>Mis-Classifications / Class-Ambiguities</font> 

Overall the BDD dataset is remarkably accurate and the class ambiguities are nuanced, however there are some consistent miscategorizations that we would like fixed in our final production dataset. Among those subtleties, a few features we should aim to correct are as follows:

---

Semi-Truck Attribute Labels


Our definition of a truck is somewhat different from the definition in the BDD dataset. To avoid this confusion, we have set up the **Semi** attribute to assign to truck labels of vehicles which **require a class A license or similar to drive.** Examples of trucks which **do not** require a Semi tag are pickups, tow trucks, RVs, trailers, etc. 

Another proper **Semi** truck type are the [**Towing Wrecker Vehicles**](https://www.google.com/search?q=towing+wreckers&source=lnms&tbm=isch&sa=X&ved=0ahUKEwiq7JCEsvfdAhWFJnwKHRfNB-gQ_AUIECgD&biw=1855&bih=990). The example above is a tow truck but should be labeled as a truck without semi for the reasons explained below in the class taxonomy.


---
| Bus Mis-Labels |
|:-----------------:|
|![bus_v_car][image11]|

---

Similar to the trucks class, our definition of a bus is more constrained. We will define a bus as only those vehicles which are large enough to have the typical bus-styled swinging door openening mechanism which attached to a turn crank. An example of a truck mis-label is shown above. In this case, we want to classify this vehicle as a bus because it likely has a door that need to swing out to open. As it is a shuttle, it is also likely to make frequent stops.


Car Mis-Labels

---

|  Example 1   |  Example 2 |  Example 3 | Example 4 |
|:-----------------:|:----------------:|:-----------------:|:-----------------:|
| ![car_antenna][image12]| ![car_mislabel0][image15]| ![car_mislabel1][image13]|  ![car_mislabel2][image14] |

---

With the exceptions to the above, our definition of cars is identical to BDD.. One small difference to note is that we will define the car bounding box to include the car itself, excluding protruding items like the car atenna mounted above the car.

In the other examples, the car classes are either a) mis-labeled altogether, or b) labeled, but missing the proper **occlusion** or **truncation** attrubutes. We must clean these instances up.


## Style Guides: Class Taxonomy

Another great feature of Scalabel is that you can easily change the classification template to whatever task suits your need. We intend to leverage these capabilities by customizing our class naming scheme. This guide will serve as the official instructions on which classes to label unless there are special circumstances.  As of now, those classes are:

- bike
- bus
- car
- traffic light (red, yellow(amber), green, N/A)
- rider (person atop a motorized vehicle or bicycle)
- truck (as well as the **Semi** truck attribute)
- train
- motor
- person (pedestrians)
- traffic sign
- trailer
- construction barrel
- construction pole
- construction cone
- construction sign

And many more soon to come.


### Class Taxonomy: Car

This category includes vans, lightweight utility vehicles, SUVs, sedans, hatchbacks, classic cars, sports cars, exotic cars, etc. This category also includes special purpose compact vehicles. i.e., Mini Shuttles, meter maids
- Note: Any type of **pickup truck** will be excluded from this category and labeled as trucks.

---

|  Cars Ex.      |
|:--------------:|
|![car1][image16]|
|![car2][image17]|
|![car3][image18]|
|![car4][image19]|
|![car5][image20]|
|![car6][image21]|
|![car7][image22]|
|![car8][image23]|

---

### Class Taxonomy: Truck

This category includes pickup trucks (light to heavy duty), trucks/straight trucks (chassis cab trucks), mail delivery trucks,  recreational vehicles (RVs), Motorized Campers, and Semi Tractors.

- Note: Chassis Cab trucks have some trailer or machinery permanently attached to the chassis of the truck. Common examples include **garbage trucks, concrete trucks**, and **box trucks**


---

|  Truck Ex.   |  Truck Ex. |
|:-----------------:|:----------------:|
| ![truck1][image24]|
| ![truck2][image25]| 
|![truck3][image26]|
|![truck4][image27]|
|![truck5][image28]|
|![truck6][image29]|
|![truck7][image30]|
|![truck8][image31]|
|![truck9][image64]|
|![truck10][image65]|
|![truck11][image66]|
|![truck12][image67]|

---

### Class Taxonomy: Semi  Truck

This sub-category corresponds to what are known as Semi Trucks. A Semi Truck has a fifth wheel coupling to attach various types of semi trailers. For our labeling purposes, these trucks should be labeled as trucks with the semi attribute whether or not the corresponding semi-trailer is attached to it. Of note, semi trucks are mechanically different from the earlier mentioned chassis cab trucks in that:
1. The trailer component is *detachable* from the chassis of the car
2. The trailer component is *articulated* . Many types of trailers (https://en.wikipedia.org/wiki/Semi-trailer)

- Note: These examples should be labeled as **truck** with the additional **Semi** attribute to distinguish from all other trucks in the category.


---

|  Semi Examples |
|:-----------------:|
|![semi1][image32]|
|![semi2][image33]|
|![semi3][image34]|
|![semi4][image35]|
|![semi][image36]|
|![semi][image37]|
|![semi7][image38]|
|![semi8][image39]|

---

### Class Taxonomy: Bus

This category includes tour buses, shuttle buses, school buses, articulated buses, trolleys, and more. Particular shuttle buses are easy to confuse with vans and some trucks. The tie breaker with these are folding doors


---

|  Bus Ex. |  Bus Ex. |  Bus Ex.  | Bus Ex. |
|:----------------:|:--------------:|:--------------:|:-------------:|
|![car_antenna][image40]|![car_antenna][image41]|![car_antenna][image42]|![car_antenna][image43]|
|![car_antenna][image44]|![car_antenna][image45]|![car_antenna][image46]|![car_antenna][image47]|

---



### Class Taxonomy: Trailer

This category includes any non-motorized vehicle generally intended to be pulled by a motorized vehicle used primarily for hauling. Exclude semi-trailers attached to Semi Trucks. Examples: camper trailer, cargo trailer, utility trailer, semi trailer (*detached*)


---

|  Trailer Ex.   | 
|:----------------:|
|![car_antenna][image48]|
|![car_antenna][image49]|
|![car_antenna][image50]|
|![car_antenna][image51]|
|![car_antenna][image52]|
|![car_antenna][image53]|
|![car_antenna][image54]|
|![car_antenna][image55]|

---

### Class Taxonomy: Construction Equipment

This category includes any machinery that is used in contruction zones and sites.
- Note: Construction Equipment is usually painted yellow or orange according to federal law. In addition, this machinery, if motorized, is not allowed to drive outside the construction zone.


---

|  Construction Equipment Ex. | 
|:----------------:|
|![car_antenna][image56]|
|![car_antenna][image57]|
|![car_antenna][image58]|
|![car_antenna][image59]|
|![car_antenna][image60]|
|![car_antenna][image61]|
|![car_antenna][image62]|
|![car_antenna][image63]|

---




## For Developers:

Here is the Json format which is supported by the training pipeline scripts currently in iPython notebooks *See e.g.*:

* **[`darkernet.py`](https://github.com/deanwebb/darknet/blob/master/darkernet.py)** (python wrapper around darknet.py)
* **`rails-reactor API` (coming-soon)**
* **`bdd-formatter ros/atm modules`** 
* **[`data_to_coco`](https://github.com/deanwebb/data_to_coco)** 


---

### Data-Format

Our dataser is compatible with the labels generated by [Scalabel](http://www.scalabel.ai/). A label [json](https://google.github.io/styleguide/jsoncstyleguide.xml) file is a list of frame objects with the fields below:

```
- name: string
- url: string
- videoName: string (optional)
- attributes:
    - weather: "rainy|snowy|clear|overcast|undefined|partly cloudy|foggy"
    - scene: "tunnel|residential|parking lot|undefined|city street|gas stations|highway|"
    - timeofday: "daytime|night|dawn/dusk|undefined"
- intrinsics
    - focal: [x, y]
    - center: [x, y]
    - nearClip:
- extrinsics
    - location
    - rotation
- timestamp: int64 (epoch time ms)
- index: int (optional, frame index in this video)
- labels [ ]:
    - id: int32
    - category: string (classification)
    - manual: boolean (whether this label is created or modified manually)
    - attributes:
        - occluded: boolean
        - truncated: boolean
        - trafficLightColor: "red|green|yellow|none"
        - areaType: "direct | alternative" (for driving area)
        - laneDirection: "parallel|vertical" (for lanes)
        - laneStyle: "solid | dashed" (for lanes)
        - laneTypes: (for lanes)
    - box2d:
       - x1: float
       - y1: float
       - x2: float
       - y2: float
   - box3d:
       - alpha: (observation angle if there is a 2D view)
       - orientation: (3D orientation of the bounding box, used for 3D point cloud annotation)
       - location: (3D point, x, y, z, center of the box)
       - dimension: (3D point, height, width, length)
   - poly2d: an array of objects, with the structure
       - vertices: [][]float (list of 2-tuples [x, y])
       - types: string (each character corresponds to the type of the vertex with the same index in vertices. ‘L’ for vertex and ‘C’ for control point of a bezier curve.
       - closed: boolean (closed for polygon and otherwise for path)
```

### BDD100K Details

Road object categories:
```
[
    "bike",
    "bus",
    "car",
    "motor",
    "person",
    "rider",
    "traffic light",
    "traffic sign",
    "train",
    "truck"
]
```
They are labeld by `box2d`.

Drivable area category is `drivable area`. There are two area types `areaType`:
```
[
    "alternative",
    "direct"
]
```

Lane marking category is `lane`. There are 8 lane styles `laneStyle`:
```
[
    "crosswalk",
    "double other",
    "double white",
    "double yellow",
    "road curb",
    "single other",
    "single white",
    "single yellow"
]
```

Both drivable areas and lane markings are labeled by `poly2d`. Please check the visulization code [`show_labels.py`](../bdd_data/show_labels.py) for examples of drawing all the labels.

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
[image16]: readme_imgs/carss1_resized.png
[image17]: readme_imgs/carss2_resized.png
[image18]: readme_imgs/car5_resized.png
[image19]: readme_imgs/carss4_resized.png
[image20]: readme_imgs/car3_resized.png
[image21]: readme_imgs/car6_resized.png
[image22]: readme_imgs/car7_resized.png
[image23]: readme_imgs/car8_resized.png
[image24]: readme_imgs/truck1_resized.png
[image25]: readme_imgs/truck2_resized.png
[image26]: readme_imgs/truck3_resized.png
[image27]: readme_imgs/truck4_resized.png
[image28]: readme_imgs/truck5_resized.png
[image29]: readme_imgs/trucks6_resized.png
[image30]: readme_imgs/truck7_resized.png
[image31]: readme_imgs/truck8_resized.png
[image32]: readme_imgs/semiss1_resized.png
[image33]: readme_imgs/semi2_resized.png
[image34]: readme_imgs/semiss3_resized.png
[image35]: readme_imgs/semiss4_resized.png
[image36]: readme_imgs/semiss5_resized.png
[image37]: readme_imgs/semi6_resized.png
[image38]: readme_imgs/semi7_resized.png
[image39]: readme_imgs/semi8_resized.png
[image40]: readme_imgs/bus1_resized.png
[image41]: readme_imgs/bus2_resized.png
[image42]: readme_imgs/bus3_resized.png
[image43]: readme_imgs/bus4_resized.png
[image44]: readme_imgs/bus5_resized.png
[image45]: readme_imgs/bus6_resized.png
[image46]: readme_imgs/bus7_resized.png
[image47]: readme_imgs/bus8_resized.png
[image48]: readme_imgs/trailer1_resized.png
[image49]: readme_imgs/trailer2_resized.png
[image50]: readme_imgs/trailer3_resized.png
[image51]: readme_imgs/trailers4_resized.png
[image52]: readme_imgs/trailers5_resized.png
[image53]: readme_imgs/trailer6_resized.png
[image54]: readme_imgs/trailer7_resized.png
[image55]: readme_imgs/trailer8_resized.png
[image56]: readme_imgs/construct-equips1_resized.png
[image57]: readme_imgs/construct-equip5_resized.png
[image58]: readme_imgs/construct-equip8_resized.png
[image59]: readme_imgs/construct-equips4_resized.png
[image60]: readme_imgs/construct-equip2_resized.png
[image61]: readme_imgs/construct-equip6_resized.png
[image62]: readme_imgs/construct-equip7_resized.png
[image63]: readme_imgs/construct-equip3_resized.png
[image64]: readme_imgs/truck9_resized.png
[image65]: readme_imgs/truck10_resized.png
[image66]: readme_imgs/truck11_resized.png
[image67]: readme_imgs/truck12_resized.png
[image68]: readme_imgs/occlusion1_resized.png
[image69]: readme_imgs/occlusion2_resized.png
[image70]: readme_imgs/occlusion3_resized.png
[image71]: readme_imgs/occlusion4_resized.png
[image72]: readme_imgs/truncation1_resized.png
[image73]: readme_imgs/truncation2_resized.png
[image74]: readme_imgs/truncation3_resized.png
[image75]: readme_imgs/truncation4_resized.png

