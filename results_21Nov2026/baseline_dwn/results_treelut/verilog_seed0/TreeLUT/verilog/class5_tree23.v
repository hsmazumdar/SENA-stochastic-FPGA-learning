module class5_tree23(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[112]) & (~i[71])) | ((~i[156]) & i[113] & i[71]) | ((~i[250]) & (~i[113]) & i[71]) | ((~i[266]) & i[112] & (~i[71])) | (i[304] & i[47] & i[156] & i[113] & i[71]) | ((~i[304]) & i[47] & i[156] & i[113] & i[71]) | (i[109] & (~i[47]) & i[156] & i[113] & i[71]) | ((~i[109]) & (~i[47]) & i[156] & i[113] & i[71]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
