from pathlib import Path

import pinocchio as pin
import example_robot_data

pinocchio_model_dir = Path(__file__).parent.parent / "third_party"
model_path = pinocchio_model_dir / "example-robot-data/robots"
mesh_dir = pinocchio_model_dir
urdf_filename = "romeo_small.urdf"
urdf_model_path = model_path / "romeo_description/urdf" / urdf_filename

# Load model
model = pin.buildModelFromUrdf(str(urdf_model_path), pin.JointModelFreeFlyer())

# Load collision geometries
geom_model = pin.buildGeomFromUrdf(model, str(urdf_model_path), pin.GeometryType.COLLISION, str(mesh_dir))

# Add collisition pairs
geom_model.addAllCollisionPairs()
print(
    "num collision pairs - initial:",
    len(geom_model.collisionPairs),
)

# Remove collision pairs listed in the SRDF file
srdf_filename = "romeo.srdf"
srdf_model_path = model_path / "romeo_description/srdf" / srdf_filename

pin.removeCollisionPairs(model, geom_model, str(srdf_model_path))
print(
    "num collision pairs - after removing useless collision pairs:",
    len(geom_model.collisionPairs),
)

# Load reference configuration
pin.loadReferenceConfigurations(model, str(srdf_model_path))

# Retrieve the half sitting position from the SRDF file
q = model.referenceConfigurations["half_sitting"]

# Create data structures
data = model.createData()
geom_data = pin.GeometryData(geom_model)

# Compute all the collisions
pin.computeCollisions(model, data, geom_model, geom_data, q, False)

# Print the status of collision for all collision pairs
for k in range(len(geom_model.collisionPairs)):
    cr = geom_data.collisionResults[k]
    cp = geom_model.collisionPairs[k]
    print(
        "collision pair:",
        cp.first,
        ",",
        cp.second,
        "- collision:",
        "Yes" if cr.isCollision() else "No",
    )

# Compute for a single pair of collision
pin.updateGeometryPlacements(model, data, geom_model, geom_data, q)
pin.computeCollision(geom_model, geom_data, 0)
