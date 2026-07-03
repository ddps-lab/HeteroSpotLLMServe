from typing import Dict, List, Optional
from collections import defaultdict
from hardware_specs import INSTANCE_SPEC

MAX_PRICE = 9876
DEFAULT_AVAILABILITY = 0

# Default OnDemand instance prices (USD per hour, us-west-2)
DEFAULT_ONDEMAND_PRICES = {
    "g4dn.xlarge":      0.526,
    "g4dn.2xlarge":     0.752,
    "g4dn.4xlarge":     1.204,
    "g4dn.8xlarge":     2.176,
    "g4dn.16xlarge":    4.352,
    "g4dn.12xlarge":    3.912,
    "g4dn.metal":       7.824,
    "g5.xlarge":        1.006,
    "g5.2xlarge":       1.212,
    "g5.4xlarge":       1.624,
    "g5.8xlarge":       2.448,
    "g5.16xlarge":      4.096,
    "g5.12xlarge":      5.672,
    "g5.24xlarge":      8.144,
    "g5.48xlarge":      16.288,
    "g6.xlarge":        0.8048,
    "g6.2xlarge":       0.9776,
    "g6.4xlarge":       1.3232,
    "g6.8xlarge":       2.0144,
    "g6.16xlarge":      3.3968,
    "g6.12xlarge":      4.6016,
    "g6.24xlarge":      6.6752,
    "g6.48xlarge":      13.3504,
    "gr6.4xlarge":      1.5392,
    "gr6.8xlarge":      2.4464,
    "g5g.xlarge":       0.42,
    "g5g.2xlarge":      0.556,
    "g5g.4xlarge":      0.828,
    "g5g.8xlarge":      1.372,
    "g5g.16xlarge":     2.744,
    "g5g.metal":        2.744,
    "g6e.xlarge":       1.861,
    "g6e.2xlarge":      2.24208,
    "g6e.4xlarge":      3.00424,
    "g6e.8xlarge":      4.52856,
    "g6e.16xlarge":     7.57719,
    "g6e.12xlarge":     10.49264,
    "g6e.24xlarge":     15.06559,
    "g6e.48xlarge":     30.13118,
    "p4d.24xlarge":     21.95764,
    "p4de.24xlarge":    27.44705,
    "p5.4xlarge":       6.88,
    "p5.48xlarge":      55.04,
    "p5e.48xlarge":     MAX_PRICE,  # No On-demand price available
    "p5en.48xlarge":    63.296,
    "p6-b200.48xlarge": 113.9328,
    "p6-b300.48xlarge": 142.416,
}


class ClusterPool:
    """
    Cluster resource and pricing manager.
    Tracks available quantity and price for each instance type.
    Used as a global object to validate the feasibility of pipeline configurations.
    """
    
    def __init__(self, available_spot_nodes: Optional[Dict[str, int]] = None,
                 spot_prices: Optional[Dict[str, float]] = None):
        """
        Initialize the cluster.

        Args:
            available_spot_nodes: Spot instance availability settings {instance_type: available_count}.
                              If None, all spot instances are initialized to 0.
            spot_prices: Spot instance price settings {instance_type: price_per_hour}.
                        If None, uses prices with the default discount rate (40%).
        """
        # Available resources (instance_type -> available_count)
        self.available_resources = defaultdict(int)
        
        # Store prices for all instances (instance_type -> price)
        self.prices = {}
        
        # Initialize all instances (OnDemand + Spot)
        for instance_type in INSTANCE_SPEC.keys():
            if not instance_type.startswith("(spot)"):
                # OnDemand instance
                self.available_resources[instance_type] = DEFAULT_AVAILABILITY
                # Set OnDemand price
                if instance_type in DEFAULT_ONDEMAND_PRICES:
                    self.prices[instance_type] = DEFAULT_ONDEMAND_PRICES[instance_type]
                else:
                    print(f"Warning: OnDemand price not found for {instance_type}")
                    self.prices[instance_type] = MAX_PRICE  # default value
            else:
                # Spot instance
                self.available_resources[instance_type] = DEFAULT_AVAILABILITY
                self.prices[instance_type] = MAX_PRICE  # default value
        
        # Update with user-defined values
        if spot_prices:
            for instance_type, price in spot_prices.items():
                if instance_type in INSTANCE_SPEC:
                    self.prices[instance_type] = price
                else:
                    print(f"Warning: {instance_type} is not in INSTANCE_SPEC")
        
        if available_spot_nodes:
            for instance_type, count in available_spot_nodes.items():
                if instance_type in INSTANCE_SPEC:
                    self.available_resources[instance_type] = count
                else:
                    print(f"Warning: {instance_type} is not in INSTANCE_SPEC")
    
    def manipulate_spot_instance_pool(self, instance_type: str, num_available: int,
                                     price: Optional[float] = None):
        """
        Set the available quantity and price for a specific instance type.

        Args:
            instance_type: The instance type to modify (e.g., "(spot)g4dn.xlarge").
            num_available: New available quantity.
            price: New hourly price (if None, the price is not changed).
        """
        if instance_type not in INSTANCE_SPEC:
            print(f"Warning: {instance_type} not found in INSTANCE_SPEC")
            return
        
        self.available_resources[instance_type] = num_available
        
        # Also update the price (if provided)
        if price is not None:
            self.prices[instance_type] = price
    
    def get_instance_price(self, instance_type: str) -> float:
        """
        Return the current price for an instance type.

        Args:
            instance_type: The instance type.

        Returns:
            Hourly price.
        """
        if instance_type in self.prices:
            return self.prices[instance_type]
        else:
            print(f"Warning: Price not found for {instance_type}")
            return MAX_PRICE
    
    def check_cluster_availability(self, stages: List[str]) -> bool:
        """
        Check whether a pipeline configuration can run on the current cluster.

        Args:
            stages: Stage configuration of the pipeline [(instance_type, layer_count), ...].

        Returns:
            Whether execution is feasible.
        """
        # Calculate the required quantity per instance type
        required_counts = defaultdict(int)
        for instance_type in stages:
            required_counts[instance_type] += 1
        
        # Check available quantity for each instance type
        for instance_type, required_count in required_counts.items():
            available = self.available_resources.get(instance_type, 0)
            if available < required_count:
                return False
        
        return True
    
    def get_resource_dict(self) -> Dict[str, int]:
        """
        Return the resource dictionary of the current cluster.

        Returns:
            {instance_type: available_count}
        """
        return dict(self.available_resources)
    
    def get_price_dict(self) -> Dict[str, float]:
        """
        Return the price dictionary of the current cluster.

        Returns:
            {instance_type: price_per_hour}
        """
        return dict(self.prices)
    
    def get_status_summary(self) -> str:
        """
        Return a summary of the cluster status as a string.
        """
        lines = ["=== Cluster Resource Status ==="]
        lines.append("\nOnDemand Instances:")
        
        # OnDemand instances
        for instance_type in sorted(self.available_resources.keys()):
            if not instance_type.startswith("(spot)"):
                available = self.available_resources[instance_type]
                if available > 0:
                    price = self.get_instance_price(instance_type)
                    lines.append(f"  {instance_type:25} : {available:4} available, ${price:.4f}/hr")

        lines.append("\nSpot Instances:")

        # Spot instances
        for instance_type in sorted(self.available_resources.keys()):
            if instance_type.startswith("(spot)"):
                available = self.available_resources[instance_type]
                if available > 0:
                    price = self.get_instance_price(instance_type)
                    lines.append(f"  {instance_type:25} : {available:4} available, ${price:.4f}/hr")
        
        return "\n".join(lines)
    
    def reset(self, available_spot_nodes: Optional[Dict[str, int]] = None,
              spot_prices: Optional[Dict[str, float]] = None):
        """
        Reset the cluster to its initial state.

        Args:
            available_spot_nodes: New Spot instance availability settings.
            spot_prices: New Spot instance price settings.
        """
        self.__init__(available_spot_nodes, spot_prices)