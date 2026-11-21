module class1_tree9(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if(((~i[278])) | (i[336] & (~i[141]) & i[84] & i[278]) | ((~i[132]) & i[27] & i[141] & i[84] & i[278]) | ((~i[45]) & (~i[27]) & i[141] & i[84] & i[278]) | (i[387] & i[180] & i[147] & (~i[84]) & i[278]) | ((~i[124]) & (~i[110]) & (~i[147]) & (~i[84]) & i[278]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
