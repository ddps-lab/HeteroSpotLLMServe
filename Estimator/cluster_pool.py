from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from hardware_specs import INSTANCE_SPEC


# OnDemand 인스턴스 기본 가격 (USD per hour)
DEFAULT_ONDEMAND_PRICES = {
    "g4dn.xlarge":      0.526,
    "g4dn.12xlarge":    3.912,
    "g4dn.metal":       7.824,
    "g5.xlarge":        1.006,
    "g5.12xlarge":      5.672,
    "g5.48xlarge":      16.288,
    "g6.xlarge":        0.8048,
    "g6.12xlarge":      4.6016,
    "g6.48xlarge":      13.3504,
    "g6e.xlarge":       1.861,
    "g6e.12xlarge":     10.49264,
    "g6e.48xlarge":     30.13118,
    "p4d.24xlarge":     21.95764,
    "p4de.24xlarge":    27.44705,
    "p5.4xlarge":       6.88,
    "p5.48xlarge":      55.04,
    "p5e.48xlarge":     9999,  # 온디맨드 가격이 존재하지 않음
    "p5en.48xlarge":    63.296,
    "p6-b200.48xlarge": 113.9328,
}


class ClusterPool:
    """
    클러스터 리소스 및 가격 관리자
    각 인스턴스 타입별로 사용 가능한 수량과 가격을 추적합니다.
    전역 객체로 사용되며, 파이프라인 구성의 실행 가능성을 검증합니다.
    """
    
    def __init__(self, available_spot_nodes: Optional[Dict[str, int]] = None,
                 spot_prices: Optional[Dict[str, float]] = None):
        """
        클러스터 초기화
        
        Args:
            available_spot_nodes: Spot 인스턴스의 가용성 설정 {instance_type: available_count}
                              None이면 모든 spot 인스턴스가 0개로 초기화됨
            spot_prices: Spot 인스턴스의 가격 설정 {instance_type: price_per_hour}
                        None이면 기본 할인율(40%)을 적용한 가격 사용
        """
        # 사용 가능한 리소스 (instance_type -> available_count)
        self.available_resources = defaultdict(int)
        
        # 모든 인스턴스의 가격 저장 (instance_type -> price)
        self.prices = {}
        
        # 모든 인스턴스 초기화 (OnDemand + Spot)
        for instance_type in INSTANCE_SPEC.keys():
            if not instance_type.startswith("(spot)"):
                # OnDemand 인스턴스
                self.available_resources[instance_type] = 9999  # 충분히 많이 사용 가능
                # OnDemand 가격 설정
                if instance_type in DEFAULT_ONDEMAND_PRICES:
                    self.prices[instance_type] = DEFAULT_ONDEMAND_PRICES[instance_type]
                else:
                    print(f"Warning: OnDemand price not found for {instance_type}")
                    self.prices[instance_type] = 9999  # 기본값
            else:
                # Spot 인스턴스
                self.available_resources[instance_type] = 0  # 기본값: 사용 불가
                # "(spot)g4dn.xlarge" -> "g4dn.xlarge"
                base_instance = instance_type[6:]  # Remove "(spot)" prefix
                if base_instance in DEFAULT_ONDEMAND_PRICES:
                    # 기본 Spot 가격은 OnDemand의 40%
                    self.prices[instance_type] = DEFAULT_ONDEMAND_PRICES[base_instance] * 0.4
                else:
                    print(f"Warning: Base OnDemand price not found for {instance_type}")
                    self.prices[instance_type] = 9999  # 기본값
        
        # 사용자 정의 값으로 업데이트
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
        특정 인스턴스 타입의 가용 수량과 가격을 설정합니다.
        
        Args:
            instance_type: 변경할 인스턴스 타입 (e.g., "(spot)g4dn.xlarge")
            num_available: 새로운 가용 수량
            price: 새로운 시간당 가격 (None이면 가격 변경 안 함)
        """
        if instance_type not in INSTANCE_SPEC:
            print(f"Warning: {instance_type} not found in INSTANCE_SPEC")
            return
        
        self.available_resources[instance_type] = num_available
        
        # 가격도 함께 업데이트 (제공된 경우)
        if price is not None:
            self.prices[instance_type] = price
    
    def get_instance_price(self, instance_type: str) -> float:
        """
        인스턴스 타입의 현재 가격을 반환합니다.
        
        Args:
            instance_type: 인스턴스 타입
            
        Returns:
            시간당 가격
        """
        if instance_type in self.prices:
            return self.prices[instance_type]
        else:
            print(f"Warning: Price not found for {instance_type}")
            return 0.0
    
    def check_cluster_availability(self, stages: List[str]) -> bool:
        """
        파이프라인 구성이 현재 클러스터에서 실행 가능한지 확인합니다.
        
        Args:
            stages: 파이프라인의 stage 구성 [(instance_type, layer_count), ...]
            
        Returns:
            실행 가능 여부
        """
        # 필요한 인스턴스 타입별 수량 계산
        required_counts = defaultdict(int)
        for instance_type in stages:
            required_counts[instance_type] += 1
        
        # 각 인스턴스 타입별로 가용 수량 확인
        for instance_type, required_count in required_counts.items():
            available = self.available_resources.get(instance_type, 0)
            if available < required_count:
                return False
        
        return True
    
    def get_resource_dict(self) -> Dict[str, int]:
        """
        현재 클러스터의 리소스 딕셔너리를 반환합니다.
        
        Returns:
            {instance_type: available_count}
        """
        return dict(self.available_resources)
    
    def get_price_dict(self) -> Dict[str, float]:
        """
        현재 클러스터의 가격 딕셔너리를 반환합니다.
        
        Returns:
            {instance_type: price_per_hour}
        """
        return dict(self.prices)
    
    def get_status_summary(self) -> str:
        """
        클러스터 상태 요약을 문자열로 반환합니다.
        """
        lines = ["=== Cluster Resource Status ==="]
        lines.append("\nOnDemand Instances:")
        
        # OnDemand 인스턴스
        for instance_type in sorted(self.available_resources.keys()):
            if not instance_type.startswith("(spot)"):
                available = self.available_resources[instance_type]
                if available > 0:
                    price = self.get_instance_price(instance_type)
                    lines.append(f"  {instance_type:25} : {available:4} available, ${price:.4f}/hr")
        
        lines.append("\nSpot Instances:")
        
        # Spot 인스턴스
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
        클러스터를 초기 상태로 리셋합니다.
        
        Args:
            available_spot_nodes: 새로운 Spot 인스턴스 가용성 설정
            spot_prices: 새로운 Spot 인스턴스 가격 설정
        """
        self.__init__(available_spot_nodes, spot_prices)