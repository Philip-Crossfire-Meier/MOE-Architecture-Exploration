import enum


class ExpertCapacityStrategy(enum.Enum):
    NONE = "None"
    FEDUS2022 = "Fedus2022"
    ZHOU2022 = "Zhou2022"

    __num_experts__ = 8
    __expert_capacity_factor__ = 1

    def __call__(self, batch_size: (int|None) = None) -> float:
        """        
        Returns the capacity of each expert based on the input size.
        Args:
            input_size (int): The size of the input data. Must be provided and greater than 0.
        """
        if self.__expert_capacity_factor__ is None or self.__num_experts__ is None:
            raise ValueError("expert_capacity_factor and num_experts must be set.")
        if batch_size is 0:
            raise ValueError("batch_size must be greater than 0.")

        if self == self.NONE:
            return float('inf')
        
        if batch_size is None:
            raise ValueError("input_size must be provided and greater than 0.")
        
        if self.__num_experts__ is None or self.__expert_capacity_factor__ is None:
            raise ValueError("num_experts and expert_capacity_factor must be set.")
        
        elif self == self.FEDUS2022:
            self.__expert_capacity_factor__ = 1.25 # Fedus (2022) suggests a buffer of 0.25
            return self.expert_capacity_Fedus2022(batch_size)
        elif self == self.ZHOU2022:
            return self.expert_capacity_Zhou2022(batch_size)

    def expert_capacity_Fedus2022(self, batch_size: int, num_cores: int = 1) -> float:
        """
        Returns the capacity of each expert, Fedus et al. (2022) version.
        Args:
            batch_size (int): The batch size.
            num_cores (int): The number of cores available for processing.
        Returns:
            float: The capacity of each expert.
        """
        tokens_per_core = batch_size / num_cores
        expert_capacity = int((tokens_per_core * self.__expert_capacity_factor__) / self.__num_experts__)
        return max(expert_capacity, 1)

    def expert_capacity_Zhou2022(self, batch_size: int) -> float:
        """
        Returns the capacity of each expert, Zhou et al. (2022) version.
        Args:
            batch_size (int): The batch size.
        Returns:
            float: The capacity of each expert.
        """
        return max((batch_size * self.__expert_capacity_factor__) / self.__num_experts__, 1)